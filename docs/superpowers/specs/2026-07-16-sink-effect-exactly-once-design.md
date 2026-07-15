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
4. Every production sink publication path, including post-run audit export,
   implements the effect protocol with deterministic prepare, convergent
   commit, and authoritative reconciliation. `flush()` is not part of any
   publication path.
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
10. Target-replacing sinks are serialized by a durable per-target stream and
    predecessor chain. Disjoint effect groups cannot plan from the same head.
11. Remote target inspection is an explicit, read-only, durably audited phase
    after reservation and before plan completion. It is neither preflight nor
    reconciliation.

## Scope

This design owns:

- the sink effect protocol and capability preflight;
- deterministic effect, artifact, and member identity;
- effect reservation, leasing, fencing, takeover, and reconciliation;
- ordered membership and overlap/rebatch recovery;
- safe plan, target, call-intent, and reconciliation evidence;
- target stream serialization and read-only inspection;
- primary, failsink, diversion, and legacy-path behavior;
- fresh, resume, follower-worker, and post-run audit-export lifecycle and
  publication behavior;
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

Two additional production lifecycle surfaces bypass the shared run-context
factory. Follower-worker startup calls sink `on_start()` directly, and
post-run audit export creates a fresh sink then calls `write()` and `flush()`
directly. Capability preflight must run at all three lifecycle boundaries
(fresh/resume, follower, export) before `on_start()` or any node-side effect.
The audit-export publication itself must move to the durable effect protocol;
preflight alone does not make its legacy write/flush sequence safe.

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

### Inspection

A post-reservation, pre-publication read of target facts needed to complete a
plan, such as an object ETag/version, a local file identity, or declared SQL
ledger capability. Remote inspection has durable call intent and result. It
cannot mutate or stage anything remotely.

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

The lineage component recursively walks `token_parents`. For a token, its
canonical structure is the tuple of
`(parent_relation.ordinal, canonical_parent_structure)` entries ordered by the
unique parent ordinal. A root has the empty tuple. This preserves fork/expand
child position and multi-parent coalesce input order without sorting identity
strings. The immutable token ID is only a collision backstop for
pathologically equal lineage structures.

Resolution is resource-bounded before reservation: maximum depth 256,
maximum 4,096 visited lineage nodes per member, maximum 1,024 parents for one
token, and maximum 64 KiB canonical serialized lineage evidence per member.
The limits are code-owned contract constants and are included in the protocol
version. A cycle, repeated/missing parent, duplicate/non-dense ordinal,
cross-run link, limit breach, or non-canonical structure fails closed as an
audit-integrity error without creating an effect.

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

### Target stream and predecessor chain

CSV, JSON, Text, S3, and Azure replace one logical target with a cumulative
snapshot. Membership deduplication alone is insufficient: two disjoint groups
could reserve concurrently, inspect the same pre-image, and each publish a
snapshot that omits the other.

Each replacing sink has a deterministic stream identity over run, sink node,
role, immutable configuration identity, and requested target template. The
stream row owns a monotonically allocated sequence and a tail effect. Effect
reservation locks the stream row, allocates the next sequence, and records the
current tail as `predecessor_effect_id` before moving the tail to the new
effect. The first effect has no predecessor. This ordering is independent of
member-lock order.

Effects may reserve membership early, but a replacing effect is not eligible
for inspection, plan completion, or commit until its predecessor is
`FINALIZED`. It inherits the predecessor's exact target/head descriptor and
builds its cumulative snapshot from that finalized chain plus its own ordered
members. Initial collision-policy resolution becomes the stream's durable
physical target; every successor must inherit it. A failed or ambiguous
predecessor blocks successors rather than allowing lost rows.

The stream-head CAS advances only when the expected predecessor and sequence
match. Concurrent disjoint groups therefore queue deterministically: group 2
cannot plan from group 0 while group 1 is unresolved, and retrying either
group preserves the same predecessor. Tests prove both groups eventually
publish one ordered cumulative target with no permanent `UNKNOWN` caused by
ordinary Elspeth contention.

## Landscape Epoch 26

### `sink_effect_streams`

Replacing targets use one stream row with:

- deterministic stream ID, run, sink node, role, and requested-target hash;
- resolved credential-safe physical target identity once the first effect's
  inspection wins;
- next sequence, tail effect ID, finalized head effect ID, and head descriptor
  hash; and
- a uniqueness constraint over `(run_id, sink_node_id, role,
  requested_target_hash)` plus run-scoped node ownership FKs.

Tail allocation and effect reservation occur in one transaction. Tail and
head references must belong to the same stream. The finalized head advances by
CAS from the effect's persisted predecessor and sequence; it cannot skip an
unfinalized predecessor.

### `sink_effects`

The group ledger stores:

- `effect_id` primary key;
- run, sink node, and closed `role` (`PRIMARY`, `FAILSINK`);
- closed state (`RESERVED`, `PREPARED`, `IN_FLIGHT`, `FINALIZED`);
- protocol version, configuration hash, ordered-membership hash, and group
  payload hash;
- reserved artifact ID and artifact idempotency key;
- redacted typed target and plan evidence, plan fingerprint, expected
  descriptor, and precondition fingerprint;
- lease owner, monotonically increasing generation, expiry, and heartbeat;
- last closed reconcile result and bounded evidence hash;
- optional durable link to the originating primary effect for failsink work;
- optional stream ID, stream sequence, and predecessor effect for replacing
  targets; a closed descriptor mode (`PRECOMPUTED`, `RESULT_DERIVED`, or
  `NO_PUBLICATION`); and
- created, updated, and finalized timestamps.

Plan/target/evidence JSON is schema-validated, size-bounded, and rejected if a
path or URI contains credentials. Validation occurs both before persistence
and immediately before publication.

Database checks mechanically enforce lifecycle completeness:

- `RESERVED`: plan/prepared/finalized fields and active lease fields are null;
- `PREPARED`: immutable plan hash/evidence, typed inspection reference,
  descriptor
  mode, prepared member dispositions, and prepared timestamp are non-null;
  active lease fields are null;
- `IN_FLIGHT`: every prepared field plus lease owner, positive generation,
  expiry, and heartbeat is non-null; finalized fields are null;
- `FINALIZED`: exact result/descriptor hash, artifact link, and finalized time
  are non-null; no active lease remains;
- `PRECOMPUTED` requires the expected descriptor in the plan;
  `RESULT_DERIVED` forbids a preclaimed result descriptor and requires an
  authoritative finalized result; `NO_PUBLICATION` requires an inherited or
  virtual exact descriptor and forbids external commit attempts; and
- a stream sequence/predecessor is present exactly for stream-bound effects,
  with a CHECK making sequence zero require no predecessor and every later
  sequence require one; composite same-stream FKs plus the tail-allocation CAS
  prove that it is the immediately preceding stream effect.

Generation is non-negative in reserved/prepared state and positive once
in-flight. Lease expiry cannot precede heartbeat. Role, state, descriptor mode,
attempt action, and reconcile result use database CHECK constraints over their
closed vocabularies.

Inspection mode is also closed: `INSPECTED` references a returned durable
inspect attempt, while `NO_INSPECTION_REQUIRED` stores a typed sentinel plan
reference and is valid only for adapter modes whose contract declares that no
initial target facts are needed. Therefore every prepared row has a mechanical
non-null inspection reference without fabricating a provider call.

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

Composite FKs prove member token/run and row/run ownership, plus the token's
row identity; a member cannot cite a token from one row/run and ordering facts
from another. Ordinals are non-negative and unique per effect, ingest sequence
is non-negative, and disposition/member-state fields use closed CHECKs.

The table stores no attempt-scoped state ID. Finalization resolves and locks
the current open state witness for each member, proves it still represents the
same token/node/input, and only then applies the generation-fenced transition.

### `sink_effect_attempts`

Every external inspect, commit, or reconcile attempt has durable intent before
I/O:

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

Effect reservation also creates one deterministic `sink_write` operation and
links it with a nullable, unique `operations.sink_effect_id` FK. Legacy/source
operations keep that column null; when non-null, `operation_type` must be
`sink_write` and run/node ownership must equal the effect. Every attempt
reserves its operation call index with the intent.

The existing operation/call audit remains authoritative for exported call
history. A returned attempt writes its redacted call row in the same
transaction that stores returned evidence. Recovery first marks an abandoned
intent `RESPONSE_LOST` and immediately writes an `ERROR` call row; it then
creates a separately intended reconcile attempt whose result creates a
distinct call. Calls do not wait for effect finalization, so an unknown or
never-finalized effect still exports honest action history. Existing
request/response hashing, payload-ref policy, and call-data redaction continue
to apply. The operation completes only when the effect finalizes; `UNKNOWN`
leaves it open with durable error/reconcile calls for operator diagnosis.
Landscape export/import, reproducibility verification, MCP diagnostics, and
web audit views include streams, effects, members, and attempts, so an
unrecovered `INTENT` is visible even before a new worker classifies it as
response-lost.

### Artifact linkage

New sink artifacts link to `sink_effect_id`, not a first member state. The
epoch-26 artifact shape supports:

- legacy epoch-25 rows linked by `produced_by_state_id`; and
- epoch-26 rows linked by `sink_effect_id`.

The linkage is exclusive. New effect finalization must use `sink_effect_id`.
`artifact_id`, `run_id`, sink node, and idempotency key must match the values
reserved on the effect. The descriptor supplied at finalization must exactly
match the persisted plan/reconciliation evidence.

This is an end-to-end contract migration, not only DDL. `Artifact`, its loader,
repository registration APIs, execution/data-flow queries, export/import
records, reproducibility checks, MCP/web serializers, AWS acceptance fixtures,
and audit views all represent the producer as an XOR of state link and effect
link. Backward reads and exports preserve epoch-25 state-linked rows exactly.
New callers cannot assume `produced_by_state_id` is non-null and cannot create
an epoch-26 sink artifact without a valid same-run/same-node effect. Exported
records carry both nullable fields plus an explicit producer kind so consumers
do not infer it from missing data.

SQLite 25 to 26 therefore performs a transactional artifacts-table rebuild
to make the state link nullable, add the effect link and exclusive check, then
adds the operation effect link and creates the stream, effect, member, and
attempt tables, constraints, and indexes. It validates epoch
25 structure and duplicate-free artifact keys before `BEGIN IMMEDIATE`,
rechecks under the lock, rolls back on any anomaly, and stamps epoch 26 only
after full verification. Exact epoch-23 databases retain the ordered
23-to-24-to-25-to-26 chain. PostgreSQL fresh schema and testcontainer probes
must match metadata. Epoch 27 is untouched.

## Global Transaction Lock Order

Epoch 26 extends the existing outcome/artifact composition order; it does not
create a parallel effect-specific order. Before implementation, every changed
caller is enumerated against one global PostgreSQL acquisition order. The
default order is:

```text
1. candidate IDs resolved by non-locking reads
2. tokens, ascending token_id
3. current node_states, ascending state_id
4. sink_effect_streams, ascending stream_id
5. sink_effects, ascending effect_id, then members by (effect_id, ordinal)
6. artifacts, ascending artifact_id
7. operations, ascending operation_id, then attempts/calls by reserved index
8. terminal token outcomes, routing, scheduler-close, and related append rows
```

Step 8 is append/validated-CAS work after the complete mutable lock set; it
does not introduce a new `FOR UPDATE` class after artifact/operation locks.

The implementation may refine the relative order of steps 6 through 8 only
after auditing all existing composition callers and updating this design's
single order everywhere. It may not change token-first, state-second, or put a
stream/effect lock ahead of a token/state lock. PostgreSQL FK key-share waits
count as lock acquisition for this rule.

Transactions resolve identifiers optimistically, begin the transaction, lock
the complete token set in sorted order, re-read/validate membership, lock the
complete current state-witness set in sorted order, and only then touch a
stream or effect. State witnesses that changed between the optimistic read and
the locked re-read cause a bounded restart from step 1, never an out-of-order
additional lock.

The existing bulk state-completion API is not evidence of step 3 merely
because its SQL updates happen to complete without deadlock. Before any epoch
26 production implementation, a prerequisite change must make that API
deduplicate the complete state set and acquire it explicitly in ascending
`state_id` order before its first pre-read or update. The primitive may use one
ordered `SELECT ... FOR UPDATE` or an ascending sequence of exact-row
`SELECT ... FOR UPDATE` statements, but its acquisition order must be visible
and testable. The effect finalizer and the composed primary-sink path must
route through that primitive after any required sorted token prelock. A
distinct-backend PostgreSQL test must pause deterministically after the first
state lock and prove the full sorted state acquisition order;
driver/executemany or planner order is never treated as the lock-order
contract.

The concrete paths obey the order as follows:

- **Reservation:** resolve membership and stream identity; lock all requested
  tokens first and any required current states second; validate ownership,
  payload hashes, and existing bindings; then insert/select-lock the stream,
  allocate its tail, lock existing effects in sorted order, and finally insert
  effect, members, and the effect-linked operation. Reservation never holds a
  stream while waiting for a token or state.
- **Plan completion and lease/takeover:** when membership/state validation is
  required, lock tokens/states before stream/effect. A narrow takeover that
  touches no token/state/stream begins at the effect step, then attempts and
  operation/call rows; it cannot later reach backward for an earlier class.
- **Attempt intent/result:** each short transaction locks the stream only when
  it must validate the current predecessor/head, then effect/member, then
  operation/attempt/call rows. No database transaction or row lock spans
  inspect, commit, or reconcile network/filesystem I/O.
- **Finalization/head CAS:** optimistically resolve all current witnesses;
  lock sorted tokens, re-resolve and lock sorted current node states, then lock
  stream and all linked primary/failsink effects in sorted order. Only then
  advance the stream head and write artifact, operation/call, routing,
  outcomes, and scheduler-close evidence in the audited terminal order.
- **Artifact mutation/delete:** an effect-linked artifact locks its effect
  before the artifact; a legacy state-linked artifact follows token, state,
  artifact order. Ordinary snapshot reads take no `FOR UPDATE` locks and never
  become a hidden reverse-order mutation.

New-stream creation uses insert-on-conflict only after token/state locks, then
locks the winning stream row. Overlapping reservations serialize at the sorted
token set; disjoint reservations can wait on the one stream insert without a
cycle because the winner never requests the loser's tokens afterward.

Real PostgreSQL tests use separate engines/connections and assert distinct
backend PIDs, bounded completion, exact winner state, and absence of deadlock
for:

- reservation versus discard/outcome composition;
- finalization versus node-state/outcome mutation;
- two overlapping and two disjoint stream reservations;
- takeover versus finalization/head CAS; and
- effect-linked and legacy artifact read/delete/mutation paths where the
  repository permits mutation.

The tests force each side to pause after its first lock so a green result
proves composed acquisition order rather than merely low contention.

## Effect State Machine

```text
unbound
  -> RESERVED     identity, members, artifact identity committed
  -> PREPARED     inspection and immutable complete plan committed by CAS
  -> IN_FLIGHT    lease owner/generation acquired by CAS
  -> FINALIZED    exact external evidence and all audit transitions committed
```

`RESERVED`, `PREPARED`, and `IN_FLIGHT` retain identity after validation,
inspection, serialization, staging, plugin, or process failure. They are
recoverable debt, not abandoned rows.

The initial reserved row has complete identity/membership/artifact facts but
nullable inspection and plan fields. After any required inspection and
effect-free preparation, one compare-and-set writes the complete immutable
plan and moves to `PREPARED`. A concurrent preparer either observes that exact
plan or compares every plan field and gets equality; divergent target,
precondition, snapshot, payload, or descriptor evidence is an integrity error.
There is no last-writer-wins plan update. `IN_FLIGHT` acquisition is rejected
until the plan-completeness checks hold and every referenced inspect attempt
has durable returned evidence.

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

Preflight is local and declarative. It validates declared Database ledger
configuration/permission requirements but does not connect to the target to
prove them, HEAD an object, read a blob, or perform DDL.

### Inspect

Once reservation is durable and any stream predecessor is finalized,
`inspect_effect` may perform the credential-safe read-only target operations
needed for an immutable conditional plan. Examples are S3 HEAD, Azure
properties, local pre-image identity/hash, and Database dialect/ledger/
permission probes. Every remote/SQL inspect has committed attempt and call
intent before it runs, a bounded timeout, and a redacted typed result stored
afterward. Inspect cannot create a table, marker, object, blob, or remote
staging resource.

Inspection is the only initial remote-precondition capture. Reconciliation is
reserved for an attempted/ambiguous publication and cannot stand in for it.
Concurrent inspectors complete one plan by CAS; evidence that would yield a
different plan fails closed instead of silently refreshing the precondition.
An adapter that declares no initial target inspection persists the typed
`NO_INSPECTION_REQUIRED` reference during plan CAS; null is never overloaded to
mean both "not inspected yet" and "inspection is unnecessary."

### Prepare

`prepare_effect` receives a restricted context, stable effect identity,
stored-ordinal rows, and (where required) a deterministic logical target
snapshot reconstructed from finalized membership plus the current effect.

Prepare may:

- validate rows and compute diversion classifications;
- render a stable target using run-start time and stable run metadata;
- consume the immutable bounded inspection evidence already stored;
- serialize and hash in memory when bounded; or
- stream into an effect-addressed, private, same-filesystem local staging
  file under explicit size/time limits.

Prepare may not perform network calls, remote reads/writes, target-side DDL,
audited provider calls, publication, or credential resolution beyond the
already-approved runtime reference. It returns typed credential-safe evidence,
never an opaque SDK handle. A durable staging reference is allowed only as an
explicit safe tagged type whose path is private, normalized, effect-addressed,
within the configured staging root, and revalidatable after process loss.

The executor stores the plan fingerprint and descriptor mode in the
reservation-to-prepared CAS. `PRECOMPUTED` plans store the exact expected
descriptor. `RESULT_DERIVED` Database plans store the full ordered input and
policy but deliberately leave the result descriptor unset until the
target-side transaction. Reprepare after process loss must reproduce the plan
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

Only the next prepared effect in the durable target stream may commit. Under a
target-scoped advisory OS lock, commit:

1. enforces a bounded lock timeout;
2. revalidates the normalized target and safe staging path;
3. verifies the predecessor/head CAS and authoritative persisted pre-image
   fingerprint, including content hash plus inode or the documented
   platform-equivalent replacement identity;
4. atomically replaces the target with staged bytes;
5. fsyncs the parent directory; and
6. verifies and returns the exact post-image descriptor.

The advisory lock reduces same-host contention but is not authority. The
durable stream predecessor, exact pre/post evidence, and atomic replace are
authority. Lock timeout fails without publication. Cross-host/shared
filesystems are supported only when they provide the documented atomic
replace, stable file identity, durable fsync, and lock semantics; otherwise
preflight rejects them rather than claiming cross-process safety.

The plan records the staged file identity because atomic replacement moves
that file identity to the public target. After an abandoned commit intent,
reconcile returns applied only when target bytes, size, and file identity are
the exact staged post-image; it returns not applied for the exact predecessor
pre-image and unknown otherwise. An unrelated writer that produces equal
bytes with a different file identity is therefore not credited as Elspeth's
publication. If the platform cannot preserve/prove the replacement identity,
the adapter is unsupported. Effect-addressed staging is reused only after
hash/path/identity validation and is cleaned after finalization or bounded
garbage-collection proof.

Append/resume incorporates the validated pre-run baseline and canonical run
snapshot. Write mode excludes the baseline. Collision-policy target selection
is deterministic and persisted before publication; a competing target claim
that invalidates the plan fails closed.

### AWS S3

Read-only S3 inspection runs after predecessor finalization and captures the
current HEAD/non-existence under durable intent. The plan contains stable
bucket/key identity, exact full logical object hash and size, that inspected
prior object version/ETag/checksum, predecessor/head identity, overwrite
policy, and safe protocol metadata. Timestamp templates use the persisted
run-start time, never retry wall time.

Commit uses a backend-native conditional `PutObject` with checksum and
credential-safe object metadata containing protocol version, effect ID, plan
hash, and content hash. First publication with `overwrite=false` requires
non-existence. Later same-run snapshots require the exact prior version/ETag.
A concurrent same-effect conditional loser performs HEAD reconciliation.

Reconcile HEADs the exact bucket/key and requires effect metadata, plan hash,
content checksum, byte size, and version evidence to match. Locally matching
effect ID never authorizes overwriting an unrelated or divergent object.
Missing target is not applied; any divergent/unverifiable target is unknown.
The stream head prevents a later disjoint group from inspecting or publishing
until this object version is finalized.

### Azure Blob

Azure performs a durably intended read-only properties/non-existence inspect
after predecessor finalization, then uses the same logical stream contract
with backend-native conditions:

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
the declaration locally without silently provisioning governance tables in an
unapproved user database. Post-reservation read-only inspection verifies the
dialect, existing ledger contract, and granted capability/permissions under a
durable SQL call intent. Provisioning occurs only through the documented
operator path or an explicitly authorized initialization mode, never inspect.

The prepared `RESULT_DERIVED` plan binds the full ordered canonical input,
target table/ledger identity, schema, duplicate/constraint policy, serializer,
and diversion policy. It does not claim an accepted-row descriptor in
advance: accepted and diverted members depend on target constraints evaluated
inside the transaction.

Commit writes the unique target-side effect marker and accepted rows in one
database transaction. The marker stores effect/plan hash, accepted member
ordinals and payload hash, and bounded diversion indices/reason hashes; it
stores no row values or credentials. A retry reads the marker and returns its
exact evidence. The marker is authoritative for the deterministic accepted /
diverted partition and the descriptor derived from the actual committed
payload. Finalization recomputes that descriptor from marker evidence and
requires exact equality. Constraint-diverted rows and accepted rows are
determined inside the transaction, so they cannot disagree with the marker.

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

## Zero-Accepted Effects

Every effect reserves one artifact identity, including a primary group whose
members all divert. For adapters that can determine the empty accepted set in
prepare, the plan uses `NO_PUBLICATION`: no external commit/reconcile attempt
is allowed, and finalization registers an artifact descriptor equal to the
finalized predecessor/head because the target is unchanged. For an initial
stream with no physical target, the descriptor is a typed virtual-empty target
with the canonical empty hash and size zero; it does not claim the file/object
exists.

Artifact metadata and every exporter/MCP/web representation surface
`publication_performed=false` plus a closed `publication_evidence_kind` of
`INHERITED` or `VIRTUAL_EMPTY`, with the predecessor effect/descriptor hash
when inherited. Consumers therefore cannot mistake a deterministic artifact
identity for proof that this effect performed a new external publication.
Ordinary committed/reconciled effects surface `publication_performed=true`
and their returned/reconciled evidence kind.

For Database, constraint diversion is result-derived and the target-side
effect marker is itself the external idempotency effect even when it records
zero accepted rows. The derived artifact describes the canonical empty
committed payload. Dataverse/Chroma groups with no valid prepared members use
the same no-publication rule. Tests cover initial empty, inherited unchanged,
all-diverted with discard, and all-diverted with failsink.

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
| After reservation, predecessor open | Stable queued identity and chain | Wait/recover predecessor; never inspect old head |
| During inspect / response loss | Stable identity plus intended read | Record response-lost and reinspect before plan CAS |
| After inspection, before prepare | Stable typed precondition evidence | Reprepare and compare |
| After plan CAS | Immutable PREPARED effect | Acquire fenced lease |
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
- concurrent disjoint groups on one replacing target allocate a predecessor
  chain, wait for the immediate head, and publish both groups without lost
  rows or ordinary-contention `UNKNOWN`;
- finalized membership cannot be rewritten;
- divergent payload for an existing member fails before I/O;
- stale lease takeover increments generation;
- stale-owner commit after takeover converges externally but cannot finalize;
- lock acquisition order is independent of effect identity;
- bounded lineage depth/node/fan-in/evidence limits fail before reservation;
- PRIMARY and FAILSINK bindings are isolated and linked; and
- no member can be rebound within the same run/sink/role.

### Reconciliation and audit

- exact pre-image, exact post-image, missing target, and divergent target;
- remote inspect intent/result is durable before plan CAS, response-lost
  inspection is retried as inspection, and divergent concurrent plans fail;
- no-inspect adapters persist the typed `NO_INSPECTION_REQUIRED` reference;
- `RESERVED` cannot acquire a lease, incomplete plans cannot become prepared,
  and complete prepared evidence is immutable;
- `UNKNOWN` never invokes commit;
- response-lost intent remains distinct from reconciliation evidence;
- bounded/redacted evidence rejects credentials and oversized provider data;
- exact returned/reconciled descriptor is required by finalization;
- repeated finalization returns the same artifact winner; and
- an artifact cannot link to a first-row/new-attempt state instead of effect;
- epoch-25 state-linked artifacts and epoch-26 effect-linked artifacts both
  load, query, export/import, reproduce, serialize through MCP/web, and pass
  AWS acceptance fixtures without nullable-link assumptions; and
- zero-accepted effects use exact no-publication/inherited/virtual semantics,
  while Database records a zero-row target marker, and every consumer surfaces
  publication-performed/evidence-kind metadata.

### Adapters

- CSV/JSON/Text use bounded streamed staging, real fsync/replace, fresh-process
  reconciliation, lock timeout, concurrent same-effect calls, cleanup, and
  unsupported-filesystem guards;
- S3 tests use contract-faithful conditional request/HEAD behavior, checksum
  and metadata equality, unrelated object refusal, same-effect loser
  reconciliation, and response loss;
- Azure tests mirror native ETag/condition/property behavior;
- Database uses SQLite and real PostgreSQL target-ledger transaction tests,
  result-derived descriptors, constraint diversion, marker collision,
  declarative preflight, read-only capability inspection, permissions, and
  unsupported DDL mode rejection;
- Dataverse and Chroma prove durable per-member partial progress and exact
  member reconciliation; and
- Chroma `skip/error` rejects before reservation/on_start/I/O.

### Schema and regressions

- fresh SQLite epoch 26 and PostgreSQL metadata parity;
- real PostgreSQL distinct-backend composed lock races for reservation versus
  outcome, finalization versus state/outcome, stream reservations, takeover
  versus finalization, and artifact mutation/read paths;
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
   artifact, predecessor, complete prepared plan, and call-intent identity
   exists.
2. Crash/retry at every caller seam produces one externally observable effect
   and one artifact identity.
3. Membership is complete, ordered by durable ingest/lineage authority, and
   independent of attempts and caller batch shape.
4. Concurrent same-effect calls converge; generation fences Landscape
   finalization; stale owners cannot mutate audit state; all composed
   PostgreSQL paths obey the one token/state/stream/effect/artifact-operation
   lock order without deadlock.
5. Concurrent disjoint replacing effects queue through one durable target
   predecessor chain and publish cumulative snapshots without lost rows.
6. Every reconciler returns only the closed three-result vocabulary with
   exact bounded evidence. `UNKNOWN` fails closed.
7. Initial remote reads occur only through durably intended read-only inspect;
   no-inspect adapters persist a typed sentinel; reserved/incomplete plans
   cannot become in-flight.
8. Primary and failsink effects are separately recoverable and durably linked;
   diverted outcomes do not terminalize before failsink durability.
9. All built-in supported modes pass adapter response-loss and exact-evidence
   tests. Unsupported modes fail before reservation or I/O.
10. New artifacts link to the complete sink effect, not the first row's state;
    old state-linked artifacts remain readable/exportable.
11. Zero-accepted effects have explicit no-publication or result-derived marker
    semantics, retain one deterministic artifact identity, and expose
    inherited/virtual publication evidence to every consumer.
12. Epoch 25 databases migrate transactionally to 26; PostgreSQL fresh schema
   is equivalent; epoch 27 remains unused.
13. Focused and broad sink/recovery suites, real PostgreSQL probes, strict
    mypy, Ruff, and pre-commit hooks pass.
