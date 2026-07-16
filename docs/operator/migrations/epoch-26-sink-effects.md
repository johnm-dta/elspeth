# Landscape Epoch 26: Recoverable Sink Effects

Landscape epoch 26 introduces durable, inspectable sink publication effects.
It replaces the unsafe gap between external I/O and internal audit evidence
with the `sink-effect-v1` reserve, prepare, lease, reconcile, commit, and
finalize protocol. Read the [Sink Effect Recovery
runbook](../../runbooks/sink-effect-recovery.md) before deploying or recovering
this release.

## Compatibility boundary

- Writable SQLite databases at exact epoch 25 migrate automatically to epoch
  26 during schema-managing initialization. The migration runs under
  `BEGIN IMMEDIATE`, validates the complete predecessor shape, rebuilds the
  changed legacy tables, creates the new effect/export tables and indexes,
  runs physical-schema and foreign-key checks, and stamps epoch 26 atomically.
  Any failure rolls the entire migration back.
- PostgreSQL does not receive runtime DDL. A schema owner must apply the
  approved epoch-26 DDL or recreate and initialize the schema before the
  DML-only runtime role starts.
- Read-only and inspection-only opens never migrate.
- Do not roll epoch-25 code back over an epoch-26 database.

The migration preserves legacy artifact, operation, and call rows. Legacy
artifacts are marked with `publication_performed=true` and
`publication_evidence_kind=legacy_returned`; they are not retroactively turned
into recoverable effects.

## New operational requirements

1. Every production sink path must declare `sink-effect-v1`, its supported
   effect modes, its supported input kinds, and all four effect methods. A
   third-party or legacy sink without the complete protocol fails closed at
   preflight. There is no direct `write`/`flush` production fallback.
2. Database sink targets need their approved target-side ledger and unique
   effect identity constraints provisioned by the target schema owner. Runtime
   credentials remain DML-only. Do not grant DDL merely to make startup pass.
3. Local file sinks require the documented locking, atomic replacement, and
   private staging semantics. Unsupported or remote filesystems fail closed
   rather than claiming exact recovery.
4. Object sinks require conditional operations and exact immutable target
   evidence. Network timeouts are reconciled; they are never treated as proof
   of absence.
5. Dataverse and Chroma use durable member sub-effects. Chroma's historical
   skip/error ambiguity is reduced to explicit, auditable member outcomes; do
   not assume every non-exception response means accepted publication.
6. Artifact producers are exclusive: an artifact is produced by exactly one
   legacy node state or one sink effect, never both. This XOR is enforced by
   the epoch-26 schema.

## Audit export cutover

Audit export now seals a bounded snapshot, spools it privately, registers
immutable content, and publishes through the effect coordinator. Enabled
exports must explicitly configure total byte/record/chunk limits, per-chunk
limits, spool cleanup budgets, a durable content store, and signing identity.
The signing key is supplied only through its secret reference; the public
signer identity and version are persisted for verification and rotation.

Before deployment, verify the spool root is local, code-owned, private, and on
a supported filesystem. Verify the content store has reference-safe garbage
collection and a retention/grace policy. Do not delete pre-cutover export files
or epoch-25 audit evidence to make the migration succeed.

## Deployment sequence

1. Stop writers and take a verified, restorable Landscape backup, including
   SQLite sidecars where applicable.
2. Record the application revision, current schema epoch, target ledger DDL
   revision, sink adapter inventory, and configured effect modes.
3. Provision PostgreSQL Landscape DDL and external database target ledgers with
   their schema-owner roles. For SQLite, leave the exact epoch-25 database
   intact for the application migration.
4. Run configuration/preflight checks. Any incomplete third-party sink,
   unsupported filesystem, missing target ledger, or unbounded export
   configuration is a deployment blocker.
5. Start one schema-managing instance, confirm epoch 26 and schema validation,
   then admit normal workers.
6. Exercise a non-production effect for each sink class and verify the full
   `RESERVED -> PREPARED -> IN_FLIGHT -> FINALIZED` history and artifact
   evidence.

If a cutover effect is uncertain, follow the recovery runbook. Do not issue a
speculative commit or manually rewrite effect state.
