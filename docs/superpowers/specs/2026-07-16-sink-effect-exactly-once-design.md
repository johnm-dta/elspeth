# Sink Effect Exactly-Once Design

Date: 2026-07-16
Status: revised after independent review; pending re-review
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
2. Effect identity is based on logical run/sink/role plus exactly one closed
   input: ordered membership facts or the immutable audit snapshot and complete
   final-manifest descriptor. It never contains an attempt-scoped node-state
   ID.
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

### Effect input kind

Every prepare request carries exactly one member of a closed union, never two
parallel optional collections and never a caller-supplied discriminator:

- `SinkEffectPipelineMembersInput(members,
  target_snapshot_members)` has non-empty current `members`; both sequences
  are frozen to tuples in dense zero-based canonical order. Its `input_kind`
  property is derived as `PIPELINE_MEMBERS`.
- `SinkEffectAuditExportSnapshotInput(snapshot_id, source_run_id,
  registry_key_hash, manifest_hash, snapshot_hash, serialization_version,
  export_format, signing_mode, signer_key_id, record_count, total_bytes,
  chunk_count, chunks, signed_manifest, reader)` has no token fields. Its
  `input_kind` property is derived as `AUDIT_EXPORT_SNAPSHOT`.

`AuditExportSnapshotChunkInput(ordinal, content_ref, content_hash, size_bytes,
record_count)` is a frozen descriptor. Ordinals are dense and non-negative;
references are credential-free; hashes are exact; sizes and record counts are
strictly positive and meet both code-owned hard limits and configured limits.
The parent validates `len(chunks) == chunk_count`, exact byte/record sums, and
tuple order before the request reaches an adapter. Export format is closed to
`JSON`/`CSV`; signing mode is closed to `UNSIGNED`/`HMAC_SHA256`. Unsigned
input requires the reserved `UNSIGNED` signer identity, while signed input
requires a non-reserved, operator-visible, credential-free `signer_key_id`.

`AuditExportSignedManifestInput(content_ref, content_hash, size_bytes,
manifest_schema, derivation_version, signature_algorithm, signature_key_id,
record_chain_algorithm, final_hash, signature)` is the public immutable
descriptor for the one final manifest. Its reference/hash
equality and lowercase-hex/positive-size rules match a chunk descriptor, but
it is not a data chunk and is excluded from `chunk_count`, `record_count`, and
`total_bytes`. `manifest_schema` is exactly
`elspeth.audit-export-manifest.v2`. `signature_algorithm` is the same closed
`AuditExportSigningMode` used by the parent; `signature_key_id` equals the
parent `signer_key_id`; and public `signature` maps exactly to the registry's
internal `signature_hex`. It is lowercase 64-hex for `HMAC_SHA256` and `None`
for `UNSIGNED`. `record_chain_algorithm` is the mode-specific literal and
`final_hash` is its lowercase 64-hex result. Descriptor `manifest_schema`,
`record_chain_algorithm`, and `final_hash` map exactly to registry
`signed_manifest_schema`, `record_chain_algorithm`, and `final_hash`.

`SinkEffectPrepareRequest(effect_id, effect_input, inspection)` accepts only
that union. Its `input_kind` is derived from `effect_input` and is never
provided by a caller. A returned `SinkEffectPlan.input_kind` must equal the
request-derived kind and the persisted effect kind.

### Audit-export snapshot

A transactionally consistent, immutable export-terminal run view materialized before any
export node, effect, operation, attempt, or call row is registered. It is a
bounded ordered chunk manifest plus immutable source-run export-terminal witness and aggregate
hash, not an unbounded copy of records in Landscape. Once its registry row
exists, every retry reads the immutable manifest/chunks and never rereads the
now-self-modified live audit tables.

The input carries a bound `RestrictedAuditExportSnapshotReader`. The frozen,
factory-only capability exposes immutable `snapshot_id`, `manifest_hash`, and
`chunk_count`, `iter_verified_chunks() -> Iterator[bytes]` for data chunks,
and one no-argument `read_verified_signed_manifest() -> bytes` for the bound
winner descriptor. It has no arbitrary-ref read method, Landscape/query
access, store credentials, or signer-key access. Before yielding each data
chunk it rechecks descriptor order, content reference, exact hash, byte size,
record count, and cumulative limits. Before returning the final manifest it
rechecks its exact ref/hash/size, canonical bytes, schema, snapshot binding,
signature metadata, and equality with the public descriptor and registry;
the factory verifies HMAC with the resolved secret before constructing the
credential-free reader. The capability is excluded from dataclass comparison
and representation; only its binding fields are compared. It is never
serialized into plans, safe evidence, or Landscape.

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

That member formula is the `PIPELINE_MEMBERS` branch. The
`AUDIT_EXPORT_SNAPSHOT` branch has zero members and uses two exact RFC 8785
components. First compute:

```text
final_manifest_identity_hash = H(C(
  "sink-effect-audit-export-final-manifest-v1",
  {
    content_hash, content_ref, derivation_version, final_hash,
    manifest_schema, record_chain_algorithm, signature,
    signature_algorithm, signature_key_id, size_bytes
  }
))
```

The object is the complete immutable `AuditExportSignedManifestInput` with no
omitted/defaulted fields: `signature` is the exact credential-safe lowercase
hex string or JSON null. Then compute the export effect ID from:

```text
H(C("sink-effect-audit-export-effect-v1", {
  export_format,
  final_manifest_identity_hash,
  input_kind: "audit_export_snapshot",
  manifest_hash,
  protocol_version,
  registry_key_hash,
  role,
  serialization_version,
  signer_key_id,
  signing_mode,
  sink_node_id,
  snapshot_hash,
  snapshot_id,
  source_run_id,
  target_config_hash
}))
```

The parent/descriptor cross-mapping is validated before hashing:
`signature_algorithm == signing_mode`, `signature_key_id == signer_key_id`,
and the mode-specific `record_chain_algorithm`, `final_hash`, and nullable
signature equal the immutable registry row and final manifest. The exact
`registry_key_hash` transitively binds exporter/public-config versions; the
formula duplicates serialization/format/signing fields so a forged registry
loader cannot hide a mismatch. `target_config_hash` binds the credential-free
publication target/configuration to the effect only. It remains absent from
snapshot identity, so the same snapshot/final-manifest winner exported to two
targets yields one snapshot and two effect IDs.

Two byte-distinct final manifests therefore cannot share an effect ID merely
because their data-chunk `manifest_hash`/`snapshot_hash` match. An identical
descriptor, including identical public signature/null and record-chain result,
converges exactly; changing any descriptor field changes
`final_manifest_identity_hash` and hence the effect ID. A same logical registry
key carrying divergent descriptor fields still fails the registry winner
exact-compare before reservation.

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
- protocol version, closed input kind (`PIPELINE_MEMBERS` or
  `AUDIT_EXPORT_SNAPSHOT`), configuration hash, ordered-membership or snapshot
  manifest hash, and group payload hash;
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

Input shape is also mechanical and portable. `sink_effects` has
`input_kind`, `required_member_ordinal`, and `required_snapshot_slot`, plus a
unique key on `(effect_id, input_kind)`. Its row CHECK has exactly two legal
forms:

```text
PIPELINE_MEMBERS:
  required_member_ordinal = 0 AND required_snapshot_slot IS NULL
AUDIT_EXPORT_SNAPSHOT:
  required_member_ordinal IS NULL AND required_snapshot_slot = 0
```

`sink_effect_members` carries an `input_kind` discriminator constrained to
`PIPELINE_MEMBERS`, a composite parent FK `(effect_id, input_kind)`, and a
unique `(effect_id, input_kind, ordinal)` key. `sink_effect_export_snapshots`
carries a discriminator constrained to `AUDIT_EXPORT_SNAPSHOT`, a required
`slot` constrained to zero, a composite parent FK, and unique keys on both
`(effect_id, slot)` and `(effect_id, input_kind, slot)`. The parent has
`DEFERRABLE INITIALLY DEFERRED` composite FKs from its required-member triple
to member ordinal zero and from its required-snapshot triple to association
slot zero. Thus pipeline effects must have member zero and cannot have an
export association; export effects must have association zero and cannot have
token members. There is no trigger that merely counts children after the fact.

Repository constructors reject the XOR before SQL, while raw-SQL tests on
SQLite and PostgreSQL prove valid parent+child transactions commit, missing
required children fail at commit, mixed children fail immediately, and
deleting either required child before commit fails the deferred parent FK.

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

### `audit_export_snapshots`

The immutable registry has one winner for:

```text
(source_run_id, exporter_version, serialization_version, format,
 signing_mode, signer_key_id, public_export_config_hash)
```

`signer_key_id` is a required operator-visible, credential-free key identity /
version when signing is enabled and a typed `UNSIGNED` sentinel otherwise. It
is never key material or a low-entropy digest of key material. Rotating a
signer must change this ID. Existing-winner lookup exact-checks it before
reuse; the same run/config with a different signer ID selects a distinct
snapshot winner or fails closed under explicit single-export policy.

The registry stores explicit `snapshot_id`, `source_run_id`, immutable
export-terminal `source_status`/`source_completed_at`, stable `exported_at`,
`registry_key_hash`,
exporter and canonical serialization versions, format, signing mode,
`signer_key_id`, `derivation_version`, `public_export_config_hash`,
chunking algorithm/per-chunk manifest-shaping limits, `record_count`,
`total_bytes`, `chunk_count`, `manifest_hash`,
`last_chunk_seal_hash`, `snapshot_hash`, `snapshot_seal_hash`, nullable
`signature_hex`, `record_chain_algorithm`, `final_hash`,
`signed_manifest_schema`, and signed-manifest hash/ref/size. The exact named,
non-partial parent index is
`CREATE UNIQUE INDEX uq_runs_export_witness ON runs(run_id, status, completed_at)`;
the ordered columns and default binary/backend collation are part of the
physical schema contract. Both fresh SQLAlchemy schema creation and the
epoch-25-to-26 migration create and verify this index before creating any
audit-export snapshot child table. The registry references those exact three
columns with a composite FK; this ordering prevents SQLite from accepting the
DDL only to raise `foreign key mismatch` on the first snapshot DML. The
snapshot CHECK allows only `completed`, `completed_with_failures`, or
`empty` with non-null completion. Both `failed` and `interrupted` are resumable:
takeover changes them to `running` and clears `completed_at`. Export refuses
either status before opening a spool, writing a chunk, or reserving an effect,
so the composite witness can never prevent a later resume.
`exported_at` is exactly the persisted immutable `runs.completed_at` witness,
not retry wall clock and never `datetime.now()`, so concurrent candidates sign
identical canonical bytes. The row contains no signing key, credentials, raw
provider body, or unbounded record payload. The operational
`runs.exported_at` written only after effect finalization is not this canonical
field and is excluded from the source snapshot.

The registry row is structurally self-consistent before any loader runs.
`MAX_AUDIT_EXPORT_SIGNED_MANIFEST_BYTES = 64 * 1024` is a code-owned schema
constant. Fresh SQLite and PostgreSQL DDL install these exact named CHECKs:

```text
ck_audit_export_snapshots_manifest_hash_hex
ck_audit_export_snapshots_snapshot_hash_hex
ck_audit_export_snapshots_snapshot_seal_hash_hex
ck_audit_export_snapshots_last_chunk_seal_hash_hex
ck_audit_export_snapshots_final_hash_hex
ck_audit_export_snapshots_signed_manifest_hash_hex
ck_audit_export_snapshots_signed_manifest_ref
ck_audit_export_snapshots_signed_manifest_size
ck_audit_export_snapshots_manifest_schema
ck_audit_export_snapshots_derivation_version
ck_audit_export_snapshots_signing_tuple
```

Each `*_hex` CHECK requires exactly 64 lowercase hexadecimal characters. Its
SQLite predicate is `length(value)=64 AND value NOT GLOB '*[^0-9a-f]*'`; its
PostgreSQL predicate is `value ~ '^[0-9a-f]{64}$'`. The ref CHECK is exactly
`signed_manifest_ref = 'sha256:' || signed_manifest_hash`. Size is
`signed_manifest_size_bytes BETWEEN 1 AND 65536`. Schema and derivation are
the literals `elspeth.audit-export-manifest.v2` and
`audit-export-derivation-v1`.

The signing-tuple CHECK has exactly two legal branches:

```text
HMAC:
  signing_mode = 'hmac_sha256'
  AND signer_key_id <> 'UNSIGNED'
  AND length(trim(signer_key_id)) > 0
  AND signature_hex is lowercase 64-hex
  AND record_chain_algorithm =
      'sha256_concat_hmac_sha256_signatures_v1'

UNSIGNED:
  signing_mode = 'unsigned'
  AND signer_key_id = 'UNSIGNED'
  AND signature_hex IS NULL
  AND record_chain_algorithm = 'sha256_concat_record_sha256_v1'
```

In the HMAC branch, “lowercase 64-hex” expands to
`signature_hex IS NOT NULL AND length(signature_hex)=64 AND signature_hex NOT
GLOB '*[^0-9a-f]*'` on SQLite, and to `signature_hex IS NOT NULL AND
signature_hex ~ '^[0-9a-f]{64}$'` on PostgreSQL. The explicit non-null term is
required because SQL CHECK expressions accept an unknown/null result.

There is deliberately no redundant registry `signature_algorithm` column;
the public descriptor/final object derives it from, and exact-compares it to,
`signing_mode`. If such a column is ever added, it must join this CHECK and
equal `signing_mode` in both branches. `final_hash` is separately hex-checked
and its semantic recomputation remains a loader responsibility.

The constraint names and dialect-specific canonical SQL join
`_REQUIRED_CHECK_CONSTRAINTS`, fresh-schema shape probes, and the epoch-26
physical manifest/fingerprint. That guard also names the supporting indexes
`uq_runs_export_witness`, `uq_audit_export_snapshots_registry_key`, and
`uq_audit_export_snapshot_chunks_terminal`. The registry index uses the exact
winner tuple shown above; the terminal index uses `(snapshot_id, ordinal,
chunk_seal_hash, cumulative_records, cumulative_bytes)` in that order to back
the deferred terminal FK. The guard also names triggers
`trg_audit_export_chunk_insert_validate`,
`trg_audit_export_snapshot_insert_seal`,
`trg_audit_export_snapshot_immutable`, and
`trg_audit_export_chunk_immutable`. PostgreSQL uses the corresponding exact
function names `fn_audit_export_chunk_insert_validate`,
`fn_audit_export_snapshot_insert_seal`,
`fn_audit_export_snapshot_immutable`, and
`fn_audit_export_chunk_immutable`; function bodies are fingerprinted with their
triggers. SQLite trigger SQL is
fingerprinted directly. Anonymous or semantically similar replacement objects
do not satisfy the required guards.

Raw SQLite and real PostgreSQL DML tests mutate each of the six named hashes,
and the HMAC signature, to uppercase, non-hex, short, and long values; supply a
syntactically valid but different signed-manifest ref; use
zero/negative/over-bound size; change the schema/derivation literal; and
exercise every mixed signing-mode/signer/null-signature/algorithm tuple. Each
row is rejected by SQL. The epoch-25-to-26 migration installs and fingerprints
the same checks in its transaction; failure injection proves rollback leaves
no named constraint/trigger/index or partial rebuilt table. Loader/reader
recomputation of hashes, HMAC, canonical bytes, and record-chain semantics is
defense in depth after this structural authority, not a substitute for it.

An `ExportReadModel` and its query adapters are connection-bound: no method
opens a second connection. Materialization opens PostgreSQL with isolation
level `REPEATABLE READ` before `BEGIN`, or issues SQLite `BEGIN` on one
dedicated connection, then performs the initial registry lookup. An exact
winner closes the read transaction immediately with zero terminal-witness or
source-record queries, then binds and verifies the winner through its stored
content-store resolver. Only on a miss does that same transaction read the
immutable export-terminal witness and enumerate every export record through
the connection-bound adapters.
PostgreSQL default `READ COMMITTED` and independent repository connections are
forbidden. A distinct-connection SQLite test proves a post-`BEGIN` insert is
excluded just as the PostgreSQL backend-PID test does.

Enumeration and canonical serialization finish into a bounded operator-owned
local spool inside that read transaction. The spool is fsynced and the DB
transaction/connection closes before any content-store read or write. Chunking
then writes bounded immutable content-addressed data objects and the separate
final-manifest object. Only after every data chunk and the final manifest are
durable does a short transaction CAS-insert or exact-compare the
immutable registry candidate. Concurrent candidates reuse an existing exact
winner; any same-key difference in descriptor, seal, signer identity,
manifest-shaping chunk policy, or canonical bytes is an audit-integrity error.
Total record/byte/chunk limits are acceptance-only: they are not in the key or
snapshot seal, and an existing winner is reusable whenever its actual totals
fit the current limits. Merely raising an acceptance limit is not divergence;
lowering one below the winner's actual total fails before effect reservation.

Candidate cleanup is reference-safe. On CAS loss or rollback, the producer
may mark only objects written by that candidate in its configured namespace;
garbage collection observes a grace period and deletes an object only after a
fresh transaction proves no winning manifest references it, including a final-
manifest reference. It never deletes
by prefix alone and never removes a winner/shared content hash. After registry
insertion, recovery uses only the registered data chunks plus final manifest
and never rereads live audit tables.

An exporter whose initial registry lookup races with another exporter keeps
the same consistent read snapshot it opened before the other export's audit
rows were committed. Both candidates therefore compute the same manifest.
This store-first boundary makes self-recursion structural; timestamp filtering
is forbidden.

### `audit_export_snapshot_chunks`

The ordered data-chunk manifest stores `(snapshot_id, ordinal)`, credential-free
`content_ref`, `content_hash`, `size_bytes`, `record_count`,
`predecessor_seal_hash`, cumulative byte/record totals, and `chunk_seal_hash`.
Every row requires a lowercase 64-hex `content_hash` and exact
`content_ref = "sha256:" + content_hash`; fresh-schema, migration, contract,
and restricted-reader tests separately reject a syntactically valid reference
whose 64-hex suffix is a different digest.

Ordinal zero stores a null predecessor that the canonical seal encoder maps to
the typed `GENESIS` value; every later row requires the exact prior row seal.
A chunk seal hashes the serialization version, snapshot ID,
ordinal, predecessor, content reference/hash, sizes/counts, and cumulative
totals. The manifest hash is the canonical hash of the ordered data-chunk
seals. The v2 final manifest is a separate content object and never occupies a
chunk ordinal or contributes to data `chunk_count`, `record_count`, or
`total_bytes`. The snapshot seal binds the registry key, immutable export-terminal witness,
manifest/last seal, actual counts, manifest-shaping chunk policy, and snapshot
hash. It does not bind acceptance-only total limits.

Chunks are inserted before the registry under a deferred chunk-to-registry FK.
Backend-specific `BEFORE INSERT` triggers are the structural authority:
inserting any chunk after its registry/seal row exists is rejected;
ordinal zero must have a null predecessor and cumulative values equal its own
counts; ordinal `n>0` must find exactly ordinal `n-1`, copy that row's
`chunk_seal_hash` as predecessor, and set cumulative records/bytes to the prior
values plus the new row. A registry insert/seal trigger checks positive count,
`min(ordinal)=0`, `max(ordinal)=chunk_count-1`, exact row count/density,
sums/final cumulative totals, and exact terminal descriptor. A deferred
composite registry-to-terminal-chunk FK binds `(snapshot_id,
terminal_chunk_ordinal, last_chunk_seal_hash, record_count, total_bytes)` to
the last chunk's unique ordinal/seal/cumulative tuple. Committing a partial or
non-dense graph is impossible.

SQLite/PostgreSQL cannot portably recompute SHA-256 in a CHECK/trigger, so the
database does not pretend to prove cryptographic content. The repository
recomputes each canonical chunk seal plus manifest/snapshot seals against
content bytes and verifies the final-manifest bytes before registry insertion.
There is no public unverified
snapshot loader: the loading API requires the winning store resolver and
recomputes all seals against bytes before returning the snapshot; the
restricted reader repeats per-chunk verification before each yield and
verifies the bound final manifest before returning it. Internal
row decoders alone are not usable snapshot capabilities. Backend mutation
guards reject `UPDATE` or `DELETE` of any sealed registry or chunk row. Raw-SQL tests cover structural
order/predecessor/totals/terminal and mutation guards; loader/reader adversarial
tests separately cover forged seal/hash/content bytes. Raw records live in the
durable bounded content store, not Landscape.

### `sink_effect_export_snapshots`

This association stores exactly one `(effect_id, snapshot_id)` for an
`AUDIT_EXPORT_SNAPSHOT` effect and is forbidden for `PIPELINE_MEMBERS`. Its
unique effect FK prevents one effect from changing snapshots. Effect identity
binds the registry key, aggregate data manifest/snapshot hashes, complete
immutable final-manifest descriptor/component, sink configuration, and
credential-free target through the exact formula above.

### Audit-export configuration and stores

Audit export has no implicit resource or identity defaults. Configuration
requires total record/byte/chunk limits, per-chunk record/byte limits, a
private spool root and cleanup age/budget, and an `AuditExportContentStore`
policy declaring a durable namespace, retention, fsync/durability capability,
and reference-safe orphan cleanup. Preflight rejects a missing/non-private
spool root, internally inconsistent limits, a content store that cannot prove
durability, or a limit above the stricter code-owned hard maximum.

Content references use the global credential-free form `sha256:<hex>`, not a
backend path or namespace. Each snapshot registry row retains the
credential-free `content_store_id` that won the CAS. The content-store resolver
must continue to resolve that ID (including replica-compatible migration) and
bind the restricted reader to it; switching current configuration may not
reinterpret the ref in a different namespace. If the winning store ID is no
longer resolvable or any chunk is unavailable, recovery fails closed instead
of materializing a new same-key snapshot. `content_store_id` is immutable
operational provenance but is not part of the byte/manifest identity; a CAS
loser on another store reuses the accessible winner and safely reclaims only
its own unreferenced replicas.

`signing_mode=HMAC_SHA256` requires both the existing secret resolver reference
and a credential-free `signer_key_id`; `UNSIGNED` requires the typed
`UNSIGNED` sentinel and forbids a secret. Rotation means deploying the new
secret with a new `signer_key_id`; reuse of an ID with different key material
is an operator error and cannot be detected by persisting a key digest because
key-derived identity is forbidden. The configured export policy explicitly
chooses either multi-version winners or single-export refusal when the signer
ID changes.

Canonical public export configuration uses the committed RFC 8785 primitive
in `contracts/hashing.py`. Its closed payloads are:

```json
{"chunking_algorithm_version":"<string>","export_format":"json|csv","exporter_version":"<string>","include_raw_error_rows":false,"per_chunk_byte_limit":1,"per_chunk_record_limit":1,"serialization_version":"<string>","signer_key_id":"<credential-free string>","signing_mode":"unsigned|hmac_sha256"}
```

for `C("audit-export-public-config-v1", payload)`, and:

```json
{"export_format":"json|csv","exporter_version":"<string>","public_export_config_hash":"<lowercase 64-hex>","serialization_version":"<string>","signer_key_id":"<credential-free string>","signing_mode":"unsigned|hmac_sha256","source_run_id":"<string>"}
```

for `C("audit-export-registry-key-v1", payload)`. Therefore:

```text
public_export_config_hash = H(C("audit-export-public-config-v1", public_config_payload))
registry_key_hash = H(C("audit-export-registry-key-v1", registry_key_payload))
```

The snapshot key is target-independent. Sink name, target/path/URI, sink public
config, secrets and secret-derived values, total acceptance limits, spool
path, content-store policy, credentials, and provider handles enter neither
hash; target/config identity belongs only in the effect ID and plan. The
exporter exact-compares every snapshot-shaping public field as well as the
hash before reusing a winner. The same snapshot exported to two targets must
therefore reuse one registry winner and reserve two target effects.
`RunLifecycle.execute_export_phase` receives the run's
`PayloadStore` and the resolved durable audit-export content store explicitly
and threads them through `export_landscape`; constructing an implicit
`RecorderFactory(db)`/default store inside the export path is forbidden.

### Canonical audit-export derivation v1 and final manifest v2

`AUDIT_EXPORT_DERIVATION_VERSION = "audit-export-derivation-v1"` fixes one
non-circular derivation order. There is no length-delimited or implicit
type-faithful encoder. Production implements one helper in
`contracts/audit_export.py`:

```python
def C(tag: str, payload: ClosedAuditExportJSON) -> bytes:
    return canonical_json({"payload": payload, "schema": tag}).encode("utf-8")
```

`canonical_json` is exactly `elspeth.contracts.hashing.canonical_json`, backed
by RFC 8785/JCS. `ClosedAuditExportJSON` permits only closed-schema objects
with string keys, ordered arrays, strings, booleans, null, and integers in
`[-9007199254740991, 9007199254740991]`. Stage validators reject extra or
missing keys, floats (including finite floats), non-finite numbers, implicit
enum instances, datetimes, bytes, tuples, sets, maps with non-string keys, and
integers outside that safe range. Callers convert enums to their exact
lowercase `.value`, hashes/signatures to lowercase 64-hex strings, content refs
to `sha256:<lowercase-64-hex>`, and instants to UTC RFC 3339 strings of the
exact form `YYYY-MM-DDTHH:MM:SS.ffffffZ` before calling `C`; no Unicode or time
normalization occurs inside `C`. Arrays preserve caller order; RFC 8785 sorts
object keys.

`H(bytes)` is `hashlib.sha256(bytes).hexdigest()` over the exact supplied bytes
and never canonicalizes again. `REF(hash)` is the exact ASCII string
`"sha256:" + hash`. The implementation and independent golden tests use these
closed stage payload schemas (the shown keys are exhaustive):

```text
public config:
  {chunking_algorithm_version, export_format, exporter_version,
   include_raw_error_rows, per_chunk_byte_limit, per_chunk_record_limit,
   serialization_version, signer_key_id, signing_mode}

registry key:
  {export_format, exporter_version, public_export_config_hash,
   serialization_version, signer_key_id, signing_mode, source_run_id}

snapshot content:
  {chunking_algorithm_version, chunks: [
     {content_hash, cumulative_bytes, cumulative_records, ordinal,
      record_count, size_bytes}
   ], record_count, serialization_version, total_bytes}

snapshot ID:
  {registry_key_hash, snapshot_hash}

chunk seal:
  {chunking_algorithm_version, content_hash, content_ref, cumulative_bytes,
   cumulative_records, derivation_version, ordinal,
   predecessor: {kind: "genesis"} |
                {hash: <lowercase-64-hex>, kind: "chunk_seal"},
   record_count, serialization_version, size_bytes, snapshot_id}

chunk manifest:
  {chunk_count, chunks: [
     {chunk_seal_hash, content_hash, content_ref, cumulative_bytes,
      cumulative_records, ordinal, predecessor_seal_hash: null|<lowercase-64-hex>,
      record_count, size_bytes}
   ], record_count, snapshot_id, total_bytes}

snapshot seal:
  {chunk_count, chunking_algorithm_version, exported_at,
   last_chunk_seal_hash, manifest_hash, per_chunk_byte_limit,
   per_chunk_record_limit, record_count, registry_key_hash,
   snapshot_hash, snapshot_id, source_completed_at, source_run_id,
   source_status, total_bytes}

export final-manifest identity:
  {content_hash, content_ref, derivation_version, final_hash,
   manifest_schema, record_chain_algorithm, signature,
   signature_algorithm, signature_key_id, size_bytes}

export effect identity:
  {export_format, final_manifest_identity_hash, input_kind, manifest_hash,
   protocol_version, registry_key_hash, role, serialization_version,
   signer_key_id, signing_mode, sink_node_id, snapshot_hash, snapshot_id,
   source_run_id, target_config_hash}
```

The corresponding tags are, respectively,
`audit-export-public-config-v1`, `audit-export-registry-key-v1`,
`audit-export-snapshot-content-v1`, `audit-export-snapshot-id-v1`,
`audit-export-chunk-seal-v1`, `audit-export-manifest-v1`,
`audit-export-snapshot-seal-v1`,
`sink-effect-audit-export-final-manifest-v1`, and
`sink-effect-audit-export-effect-v1`. `predecessor.kind="genesis"` is the only
GENESIS representation; it has no `hash` key. A later predecessor must use the
second exact object shape. The snapshot-content chunks intentionally omit refs,
snapshot ID, and seals; content hash plus size/order/totals defines content.

Every candidate executes these steps before the registry compare-and-set
(CAS):

1. Compute `public_export_config_hash` and `registry_key_hash` with the first
   two schemas.
2. Serialize source records in stable order. Each record without its public
   `signature` field is RFC 8785 UTF-8. In `HMAC_SHA256`, add the current public
   per-record `signature = HMAC-SHA256(key, unsigned_record_bytes).hexdigest()`;
   in `UNSIGNED`, omit `signature`. Frame each final data-record object as its
   exact RFC 8785 bytes followed by `b"\n"`. `exported_at` is the immutable run
   completion witness. Exporter-owned fields are reserved from caller payloads.
3. Preserve the current signed record-chain semantics:
   `final_hash = SHA256(concat(ASCII(record.signature) for records)).hexdigest()`
   and `record_chain_algorithm = "sha256_concat_hmac_sha256_signatures_v1"`.
   For v2 unsigned exports, use the deterministic replacement
   `final_hash = SHA256(concat(ASCII(SHA256(unsigned_record_bytes).hexdigest())
   for records)).hexdigest()` and
   `record_chain_algorithm = "sha256_concat_record_sha256_v1"`. The manifest is
   not a data record and never enters either chain.
4. Chunk only complete framed data records. For each exact `chunk_bytes[i]`,
   compute lowercase `content_hash[i]` and exact `content_ref[i] =
   REF(content_hash[i])`. Data chunks exclude the final manifest.
5. Compute `snapshot_hash`, then `snapshot_id`, then dense chunk seals from the
   exact schemas above. Compute `manifest_hash` from the ordered sealed chunk
   descriptors, then `snapshot_seal_hash`. Acceptance-only total limits,
   target/effect identity, content-store ID, final-manifest descriptor, and
   signature remain excluded from all earlier stages.
6. Construct the exact final-manifest core object below, except `signature`.
   The signing body is
   `C("audit-export-final-manifest-signing-body-v2", core_object)`. For
   `HMAC_SHA256`, compute lowercase `signature_hex =
   HMAC-SHA256(key, signing_body).hexdigest()`; for `UNSIGNED`, require
   `signature_hex is None` and `signer_key_id == "UNSIGNED"`.
7. Add exactly one `signature` key whose value is `signature_hex` or JSON null,
   encode the object once with RFC 8785, with no leading/trailing whitespace or
   newline, and define
   `signed_manifest_size_bytes = len(signed_manifest_bytes)`,
   `signed_manifest_hash = SHA256(signed_manifest_bytes).hexdigest()`, and
   `signed_manifest_ref = REF(signed_manifest_hash)`.

The final object has this exhaustive v2 field set and types:

```json
{"chunk_count":1,"derivation_version":"audit-export-derivation-v1","export_format":"json|csv","exported_at":"YYYY-MM-DDTHH:MM:SS.ffffffZ","final_hash":"<lowercase 64-hex>","hash_algorithm":"sha256","last_chunk_seal_hash":"<lowercase 64-hex>","manifest_hash":"<lowercase 64-hex>","record_chain_algorithm":"sha256_concat_hmac_sha256_signatures_v1|sha256_concat_record_sha256_v1","record_count":1,"record_type":"manifest","registry_key_hash":"<lowercase 64-hex>","run_id":"<source_run_id>","schema":"elspeth.audit-export-manifest.v2","signature":"<lowercase 64-hex>|null","signature_algorithm":"hmac_sha256|unsigned","signature_key_id":"<credential-free ID|UNSIGNED>","snapshot_hash":"<lowercase 64-hex>","snapshot_id":"<lowercase 64-hex>","snapshot_seal_hash":"<lowercase 64-hex>","source_completed_at":"YYYY-MM-DDTHH:MM:SS.ffffffZ","source_status":"completed|completed_with_failures|empty","total_bytes":1}
```

Angle brackets and `|` above are schema metavariables, not literal output;
golden vectors contain one concrete value for each mode. Count/size fields are
safe non-negative integers, with `chunk_count`, `record_count`, and
`total_bytes` positive for the required run record.

`signature_algorithm` equals the internal `signing_mode` value exactly,
`signature_key_id` equals `signer_key_id`, and public `signature` equals
internal `signature_hex`. Signed and unsigned manifests contain the same keys;
only the three signing values and record-chain algorithm differ. The signed
preimage omits only `signature`. Self-address fields
`signed_manifest_hash`, `signed_manifest_ref`, `signed_manifest_size_bytes`
(and aliases such as `content_hash`, `content_ref`, or `size_bytes` for the
manifest itself) are absent from both the signing preimage and final object.
They live only in the registry and `AuditExportSignedManifestInput`, preventing
self-reference. The reader rejects a final object with any extra/missing key,
wrong canonical bytes, wrong mapping, or wrong descriptor binding.

Transport never changes those bytes. JSON/JSONL target bytes are exactly
`b"".join(verified_data_chunks) + signed_manifest_bytes`: every data record
frame already ends in `b"\n"`, while the final manifest is the last record at
EOF and has no trailing newline. Its descriptor/hash/size cover only those
canonical manifest bytes, and JSON reconcile hashes/compares that exact target
suffix and complete target byte string. CSV writes the identical bytes, also
without a newline, at the literal reserved relative path
`audit_manifest.v2.json`. No record type or generated CSV filename may claim
that path or any case-folding alias; exact-tree reconciliation rejects aliases,
duplicates, and case collisions.

The registry stores the derivation version, detached `signature_hex` (nullable
only for `UNSIGNED`), record-chain result/algorithm, literal signed-manifest
schema, and signed-manifest
hash/ref/size. Its exact-winner comparison checks every derived field and byte
stream. All derivations remain independent of sink name, target, effect ID,
and `content_store_id`; the effect identity separately binds target
configuration. Production golden tests hard-code literal expected UTF-8 bytes,
SHA-256/HMAC hex, and refs for public config, registry key, snapshot content,
snapshot ID, every chunk seal including GENESIS, chunk manifest, snapshot seal,
signing body, and both signed and unsigned final-manifest bytes/address. Those
expectations must be written as literals and must not call `C`, the production
schema builders, or the production derivation helper.

A barriered concurrency test opens distinct repeatable-read snapshots, lets
both contenders finish record serialization, chunk storage, every derivation,
and signed-manifest storage, then releases both into registry CAS. Before and
after CAS it asserts byte-for-byte equality of record/chunk bytes, registry
key, snapshot hash/ID, every content hash/ref and chunk seal, manifest hash,
snapshot seal, signing body/signature, and final signed-manifest bytes. One row
wins; the loser exact-compares and reuses it.

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
TUI `explain_screen.py`/`lineage_view.py`/`types.py`, and audit views all
represent the producer as an XOR of state link and effect link. The direct S3
publication proof in `web/aws_ecs_acceptance.py` must use the effect protocol
or be explicitly removed from the production-call inventory; it may not remain
a write/flush exception. Backward reads and exports preserve epoch-25
state-linked rows exactly.
New callers cannot assume `produced_by_state_id` is non-null and cannot create
an epoch-26 sink artifact without a valid same-run/same-node effect. Exported
records carry both nullable fields plus an explicit producer kind so consumers
do not infer it from missing data.

SQLite 25 to 26 therefore performs transactional `artifacts` and `operations`
table rebuilds to make the artifact state link nullable, add both effect links,
and install the exclusive checks before creating the stream, effect, member,
snapshot, and attempt objects. The live constructor invokes
`_migrate_sqlite_schema()` before compatibility validation, so that method is
the only migration dispatcher: its bounded loop grows from two to three steps
and walks exactly 23 -> 24 -> 25 -> 26. `_sync_sqlite_schema_epoch()` remains
the post-validation fresh-schema stamp/guard and never dispatches a populated
predecessor migration. The older 23->24 step treats a peer already at 24, 25,
or 26 as success both after predecessor-validation failure and after acquiring
the write lock; the 24->25 step likewise accepts a peer already at 25 or 26.

The 25->26 step follows the epoch-23-to-24 raw-connection discipline. It
validates the exact predecessor before raw connection checkout, sets
`PRAGMA foreign_keys=OFF` and verifies zero before `BEGIN IMMEDIATE`, then
rechecks the epoch under the writer lock. A peer-completed epoch 26 rolls back
the empty transaction and returns; any other predecessor change fails. The
transaction snapshots dependent DDL, rebuilds populated `operations` and
`artifacts` with explicit-column `INSERT SELECT`, and restores their exact
indexes/triggers while preserving every operation primary key referenced by a
`calls.operation_id` child. It then creates the new epoch-26 objects, runs
`PRAGMA foreign_key_check`, validates the complete physical manifest, stamps
26, and commits. Every failure rolls back all rebuilt and new objects. The
finally path restores and verifies `PRAGMA foreign_keys=ON` on every exit; a
rollback, restoration, or close failure marks the physical connection
uncertain and invalidates it before surfacing the primary/cleanup error.

Migration tests preserve an operation plus its child call byte-for-byte, race
two openers through the writer lock, inject failures during each rebuild/new-
object phase, compare the full pre/post `sqlite_schema` to prove no partial
objects, and cover rollback plus foreign-key-restore/close failure and
invalidation. Exact epoch-23 databases retain the ordered
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
- the configured mode appears in `supported_effect_modes` and has an exact
  commit/reconcile implementation;
- the lifecycle's required input kind independently appears in
  `supported_effect_input_kinds`; mode support never implies input support;
- required target-side ledger permissions/capabilities are configured;
- path and URI policies are safe and credential-free after redaction; and
- configured resource bounds are valid.

Unsupported third-party plugins, Chroma `skip`/`error`, and Database modes or
dialects without required transactional behavior fail here with specific
remediation.

Preflight is local and declarative. It validates declared Database ledger
configuration/permission requirements but does not connect to the target to
prove them, HEAD an object, read a blob, or perform DDL.

The collection validator receives `required_input_kind` explicitly and never
derives it from mode or sink config. Shared fresh/resume and follower-worker
boundaries require `PIPELINE_MEMBERS`; the fresh post-run export boundary
requires `AUDIT_EXPORT_SNAPSHOT`. A sink may support one without the other,
and the wrong boundary rejects it before lifecycle, reservation, credentials,
or I/O.

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
inspection, and exactly one closed effect input. For `PIPELINE_MEMBERS`, that
input contains stored-ordinal current rows and the deterministic logical target
snapshot reconstructed from finalized membership plus the current effect. For
`AUDIT_EXPORT_SNAPSHOT`, it contains only the immutable registry metadata,
bounded descriptors, and its bound restricted reader.

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

The adapter must copy the request-derived input kind into its plan. The
coordinator rejects a mismatch with the persisted effect before plan CAS. An
audit-export adapter may consume data chunks only through
`iter_verified_chunks()` and the one final manifest only through
`read_verified_signed_manifest()`; it cannot replace descriptors, request an
arbitrary content ref, query Landscape, access signer credentials, or serialize
the capability into evidence.

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

### Post-run audit export

Audit export never synthesizes pipeline tokens. Before registering an export
node/effect/operation/call, the export coordinator materializes or reuses the
typed `AUDIT_EXPORT_SNAPSHOT` registry winner. The effect has zero token
members, one snapshot association, and exactly the Task 6 identity over the
immutable snapshot, complete final-manifest descriptor/component, and safe
sink target/configuration. This coordinator consumes that identity and neither
derives nor redefines it. It otherwise uses the same inspect, immutable plan,
attempt, lease, commit, reconcile, and finalization lifecycle.

JSON audit export replays the ordered verified data chunks and then writes the
exact bytes from `read_verified_signed_manifest()` as the one final record; it
rejects a missing, duplicate, non-final, or tampered manifest, and retries never
enumerate live Landscape rows. CSV multifile export uses a dedicated
create-only directory-bundle adapter: prepare reconstructs the complete bundle
into a private effect-addressed sibling staging directory containing the data
files plus the exact verified v2 audit manifest at the reserved literal path
`audit_manifest.v2.json`. The canonical directory-bundle manifest binds that
audit-manifest file like every other file, and the plan binds every relative file hash/size plus
one aggregate bundle hash. Prepare fsyncs
each completed regular file and the staging directory before plan completion.
Commit publishes
on Linux only through `renameat2(AT_FDCWD, staging, AT_FDCWD, target,
RENAME_NOREPLACE)` via a checked libc/syscall wrapper. Preflight requires
Linux, the syscall/flag, a read-only `statfs` result in the explicit supported
local-filesystem allowlist, same-device sibling staging/target parents, and a
path whose existing components are non-symlinks. Before snapshot reservation,
an engine-owned bounded private sibling probe exercises both successful
`RENAME_NOREPLACE` and `EEXIST`, then file/directory/parent fsync and cleanup;
stale probe names are bounded and cleaned on the next preflight. This is not a
plugin call or target publication and is the sole exception to declarative
capability preflight. Any probe failure refuses the export. `EEXIST`
never falls back to replacement: it enters exact-tree
reconciliation. Other platforms/filesystems fail before snapshot reservation;
any future alternative needs its own immutable-generation plus atomic-pointer
design.

Exact-tree reconciliation opens relative paths beneath directory FDs without
following symlinks and rejects every extra or missing entry, non-regular file,
symlink, path escape, duplicate/case-colliding name, changed canonical
manifest, hash, or size. Only an identical manifest and file identity is
exact; a legacy/unrelated target directory is a divergent collision. After a
successful rename the adapter fsyncs the bundle directory and parent before
return. Fault tests cover target creation between inspect and rename,
`EEXIST`, crash before rename, after rename/before either fsync, after bundle
fsync, and after parent fsync. The artifact descriptor names one bundle and
aggregate hash. Direct
`_export_csv_multifile`, sink `write()`, and sink `flush()` publication are
removed. Filesystems without the required directory replacement/identity
semantics and remote sink modes without an exact bundle primitive fail
preflight before snapshot reservation or I/O.

Tests force crashes before snapshot registration, after registry insertion,
after external publication, and after finalization; concurrent exporters must
reuse one registry/effect winner. They also mutate audit tables after snapshot
registration and prove recovery still emits the original manifest, verify the
immutable export-terminal witness, and prove no export row can recursively enter its own
snapshot.

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
- Audit-export snapshots have configured total record/byte/chunk limits and
  per-chunk record/byte limits; consistent-read enumeration writes only to a
  bounded local spool, and chunk-store I/O starts after the DB transaction
  closes.
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
| Before audit snapshot registry | No export effect; possible unreferenced content chunks | Rematerialize from an immutable export-terminal consistent read; bounded cleanup removes orphans |
| After audit snapshot registry, before export effect | Immutable manifest/chunks | Reuse registry winner; never reread live audit tables |
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
- PRIMARY and FAILSINK bindings are isolated and linked;
- no member can be rebound within the same run/sink/role;
- pipeline-member/export-snapshot XOR rejects every mixed or empty shape; and
- concurrent exporters reuse one snapshot/effect winner without synthetic
  tokens, even when later export audit rows change the live database;
- audit-export effect identity binds every immutable final-manifest descriptor
  field plus the target config: identical complete descriptors converge,
  every valid field difference changes the effect ID, and divergent
  descriptors under one registry key fail exact comparison;
- frozen prepare inputs reject mutation, dense-order/count/sum violations,
  unsupported formats/signing modes, mismatched reader bindings, and attempts
  to serialize the reader capability;
- an existing registry winner performs zero source-record queries, while a
  divergent candidate for the same registry key fails closed; and
- signed concurrent candidates produce byte-identical output from the stable
  immutable export-terminal timestamp and exact signer/config identity.

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
- Chroma `skip/error` rejects before reservation/on_start/I/O;
- audit JSON replays only registered data chunks followed by the bound v2 final
  manifest, preserving current per-record HMAC signatures and manifest-last
  semantics;
- audit JSON rejects missing, corrupt, reordered, descriptor-mismatched, and
  oversized chunks before adapter-visible bytes, plus tampered, missing,
  duplicate, or non-final manifests;
- exporter, JSON, and CSV tests pin independent literal RFC 8785/HMAC golden
  bytes for both signed and unsigned manifests, require the public/internal
  signature mapping, and verify the adapter output bytes/tree exactly;
- CSV multifile uses Linux `renameat2(RENAME_NOREPLACE)` with a forced target
  creation race, exact-tree extra/missing/symlink/path-escape/hash checks,
  legacy-directory collision, every fsync crash seam, exact-existing
  convergence, and unsupported-platform/filesystem guards; and
- an AST/caller inventory proves every production `sink.write()`/`flush()`
  call, including follower, export, and AWS acceptance surfaces, has either
  migrated to effects or is an explicitly non-production compatibility test.

### Schema and regressions

- fresh SQLite epoch 26 and PostgreSQL metadata parity;
- real PostgreSQL distinct-backend composed lock races for reservation versus
  outcome, finalization versus state/outcome, stream reservations, takeover
  versus finalization, and artifact mutation/read paths;
- exact 25-to-26 migration dispatched from `_migrate_sqlite_schema()`,
  23-to-24-to-25-to-26 ordered migration, populated operation+child-call and
  artifact preservation, rollback/no-partial-object comparison, concurrent
  opener short-circuits through epoch 26, FK restore/close invalidation, lock
  contention, duplicate/malformed predecessor refusal, and reopen;
- real PostgreSQL repeatable-read snapshot winner and concurrent post-snapshot
  audit-row exclusion, with no DB transaction spanning chunk-store I/O;
- SQLite distinct-connection consistent-snapshot exclusion and a source-query
  spy proving the existing-winner fast path performs no enumeration;
- raw-schema and lifecycle tests reject `failed`/`interrupted` witnesses before
  spooling, prove takeover can still set `running`/clear `completed_at`, then
  allow export only after resumed execution reaches an immutable export-terminal
  status;
- raw-SQL SQLite and PostgreSQL commit/delete proofs for both input XOR forms,
  snapshot terminal/composite FKs, immutable mutation guards, dense chunk
  predecessor-reference/cumulative-total validation, and partial-graph
  rollback, with separate loader/reader cryptographic-forgery tests;
- raw SQLite and real PostgreSQL registry DML, on fresh and migrated schemas,
  enforce the six lowercase hash fields, exact signed-manifest ref/hash,
  bounded size, literal schema/derivation version, and closed signing tuple;
- content-store candidate rollback/orphan cleanup proves shared and winner data
  chunks/final manifests survive, and only unreferenced candidate-owned objects
  age out;
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
   or exact audit-export snapshot association, artifact, predecessor, complete
   prepared plan, and call-intent identity exists.
2. Crash/retry at every caller seam produces one externally observable effect
   and one artifact identity.
3. Pipeline membership is complete, ordered by durable ingest/lineage
   authority, and independent of attempts and caller batch shape; audit-export
   input has zero token members and exactly one immutable snapshot whose full
   final-manifest descriptor is bound into effect identity.
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
14. Audit export materializes one bounded immutable export-terminal
    repeatable-read snapshot
    before export audit rows, closes the DB transaction before chunk-store I/O,
    binds the complete final-manifest descriptor and credential-free target
    identity, reuses one concurrent snapshot winner, and recovers JSON/CSV
    publication without live-table rereads or direct `write()`/`flush()`.
