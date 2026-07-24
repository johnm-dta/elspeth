# Current State — ELSPETH

**Checkpoint:** 2026-07-24
**Release branch:** `release/0.7.2`
**Release-prep issue:** `elspeth-64c319bf4d`
**Release milestone:** `elspeth-6343920a47`

## The Bet Right Now

**Prepare 0.7.2 as a distinct maintenance release, prove one unchanged
candidate, then hand the operator the signing and publication boundary.**

Commit `720d44133` is the semantic 0.7.1 release point. The current release
branch contains the subsequent production-path hardening; those changes must
not remain folded into the historical 0.7.1 notes.

## Current Release State

- The root package metadata and lockfile identify 0.7.2.
- Current release labels, container examples, website footers, and release
  documentation indexes identify the 0.7.2 line.
- `CHANGELOG.md` preserves the 0.7.1 session cutover at epoch 35 and assigns the
  epoch-36 blob-deletion cleanup boundary to 0.7.2.
- `SESSION_SCHEMA_EPOCH` is 36, guided checkpoint schema is 10, and
  `SQLITE_SCHEMA_EPOCH` is 29. An upgrade from 0.7.1 recreates a stale session
  store; a Landscape store already at epoch 29 remains current.
- No 0.7.2 tag or final release candidate has been published.

## Release Gates

- Resolve autonomous failures in the complete local release suite and bind the
  result to one unchanged release SHA.
- Re-run PostgreSQL, packaging/container, and live AWS acceptance against that
  final SHA; older Plan 12 evidence does not transfer to the moved branch tip.
- Complete the operator-held trust-tier judgment signing and regenerate the
  fingerprint baseline through the supported tooling. Do not bypass signature
  verification or hand-edit the baseline.
- Tag, push, publish images, and create the GitHub release only after every
  required gate passes on the same candidate.

## Next Session, Start Here

1. Inspect `elspeth-64c319bf4d` and the exact release-branch SHA.
2. Resume the first incomplete release gate without changing the candidate.
3. Keep signing and external publication operator-controlled.
