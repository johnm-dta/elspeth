# Current State — ELSPETH

**Checkpoint:** 2026-07-14
**Release branch:** `release/0.7.1`
**Runtime-readiness integration branch:** `feat/aws-ecs-program`
**Release milestone:** `elspeth-6343920a47`

## The Bet Right Now

**Complete and integrate the 0.7.1 AWS ECS runtime-readiness programme, then
run the owned release closeout against one unchanged candidate.**

The programme branch now contains the reviewed `release/0.7.1` tip plus the
completed PostgreSQL, S3, Bedrock, Cognito, telemetry, packaging, and deployment
slices. The release branch itself does not contain that programme until the
coordinator completes Plan 12 and performs the planned final fast-forward.

## Current Release State

- The root package metadata and lockfile identify 0.7.1.
- Current release labels, container examples, website footers, and release
  documentation indexes identify the 0.7.1 line.
- `CHANGELOG.md` contains the release branch's Composer notes and the integrated
  schema-cutover correction; Plan 12 still owns the final AWS programme entry.
- `SESSION_SCHEMA_EPOCH` is 30 and `SQLITE_SCHEMA_EPOCH` is 28. Epoch 29 owns
  guided schema 8 and durable operation fencing; epoch 30 closes the
  `quota_exceeded` terminal failure code used by stable HTTP 413 fork replay.
  The integrated
  candidate requires a two-database cutover from older schemas. Because ELSPETH
  is pre-1.0, neither database is migrated in place: archive/export when
  required, uninstall, recreate both stale stores, and reinstall.
- No 0.7.1 tag or final release candidate has been cut.

## In Flight

- **Plan 12 — final integration closeout** (`elspeth-05396fed38`) is the sole
  remaining milestone step.
- All 19 prerequisite steps, including universal web plugin policy, Bedrock
  Guardrail shields, PostgreSQL doctor proof, and packaging/deployment, are
  closed with program-branch commit anchors.
- `feat/aws-ecs-program` must remain the evidence-bound integration surface
  until its final gates complete; do not describe that work as released from
  the current `release/0.7.1` checkout.

## Release Blockers

- Complete the remaining Filigree critical path and close all implementation
  prerequisites with commit anchors.
- Run the full Plan 12 local, hosted-CI, trust, live AWS, rollback, evidence,
  and teardown gates against one unchanged candidate SHA.
- Fast-forward `release/0.7.1` to that accepted candidate only after the gates
  pass; then rebuild and verify the durable release artifacts under the
  release-owner workflow.
- Record final approval, artifact provenance, and operator-owned evidence
  before publishing a tag or image.

## Next Session, Start Here

1. Resume Plan 12 on the reconciled programme branch and freeze one candidate.
2. Keep release-branch publication claims separate until final acceptance.
3. Refresh the 0.7.1 changelog against the accepted candidate before the final
   release fast-forward.
