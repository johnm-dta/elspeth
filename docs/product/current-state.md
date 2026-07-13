# Current State — ELSPETH

**Checkpoint:** 2026-07-14
**Release branch:** `release/0.7.1`
**Runtime-readiness integration branch:** `feat/aws-ecs-program`
**Release milestone:** `elspeth-6343920a47`

## The Bet Right Now

**Complete and integrate the 0.7.1 AWS ECS runtime-readiness programme, then
run the owned release closeout against one unchanged candidate.**

The release branch contains the post-0.7.0 Composer work described in the root
changelog. The separate AWS programme branch carries additional PostgreSQL,
S3, Bedrock, Cognito, telemetry, packaging, and deployment work. Those branches
have diverged; the AWS programme is not part of the release branch until its
coordinator completes the planned final fast-forward.

## Current Release State

- The root package metadata and lockfile identify 0.7.1.
- Current release labels, container examples, website footers, and release
  documentation indexes identify the 0.7.1 line.
- `CHANGELOG.md` records only changes present on `release/0.7.1`; it does not
  claim the unmerged AWS programme as shipped.
- `SESSION_SCHEMA_EPOCH` is 27. `SQLITE_SCHEMA_EPOCH` remains 22, so an upgrade
  from 0.7.0 requires a session-database recreation but not another Landscape
  reset.
- No 0.7.1 tag or final release candidate has been cut.

## In Flight

- **Plan 15B — universal web plugin policy** (`elspeth-0674a06468`) is the
  current P1 step on the programme critical path.
- **Plan 15C — Bedrock prompt and content Guardrail shields** follows Plan 15B.
- PostgreSQL doctor proof, packaging and deployment documentation, and final
  integration closeout remain downstream work.
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

1. Read the live Filigree milestone and resume the current critical-path owner.
2. Keep release-branch documentation claims separate from unmerged programme
   claims.
3. After programme integration, refresh the 0.7.1 changelog against the final
   release SHA and rerun every release gate.
