# Current State — ELSPETH

**Checkpoint:** 2026-07-08
**Branch:** `release/0.7.0`
**Release PR:** #86, `release/0.7.0` → `main`

## The Bet Right Now

**Ship the 0.7.0 line to merge-ready, minus the operator signing stage.**

The 0.7.0 release is the active delivery focus. It packages the LLM-primary
guided Composer, first-run tutorial recut, document-ingestion plugins, CLI/TUI
operator refresh, local Composer user/email-verification flow, public website,
release documentation refresh, and the release hardening body recorded in the
root changelog.

The longer-running **Web hardening to GA** bet remains on the roadmap, but the
current checkout is in release-closeout mode: finish PR #86, preserve the
operator-only signing boundary, and avoid adding new scope that does not move
the release toward merge.

## Current Release State

- Public docs have been refreshed for 0.7.0 across `CHANGELOG.md`, `README.md`,
  `ARCHITECTURE.md`, `SECURITY.md`, `GOVERNANCE.md`, `SUPPORT.md`, and
  `docs/release/`.
- Implemented plans, superseded specs, generated review sidecars, and other
  work-product docs have been removed from the tracked active docs tree. A
  maintainer may keep local copies under ignored `docs-archive/`; git history is
  the public provenance record.
- `docs-archive/` is intentionally local-only and ignored. It is not sensitive,
  but it is not part of the public repository.
- Local release verification is green for the currently exercised lint, type,
  contract, unit, and frontend E2E checks recorded in PR #86.
- GitHub CI is green except for the expected operator-owned Static analysis
  failure caused by signed trust-tier allowlist drift.

## In Flight

- **PR #86 release closeout** — docs are current, PR text records scope,
  verification, archive policy, CodeQL/E2E fixes, and the remaining
  operator-only stage.
- **Signed trust-tier allowlist repair** — tracked as
  `elspeth-2670a38693`, assigned to `operator`. Agents should diagnose and
  report drift, but must not mint or rotate signed allowlist metadata.
- **Final live-judge verification** — owed after the operator signing process,
  using the real judge-signature key material and live LLM path.

## Open Questions / Blockers

- Operator must repair/sign the trust-tier allowlist with release key material.
- Operator must rerun Static analysis / aggregate CI after signing.
- Operator must run the final live-judge end-to-end path before merge.
- Release approval still needs final provenance in the public release materials.
- Security disclosure and supply-chain artifact evidence remain publication
  readiness items unless explicitly deferred in the release record.

## Last Checkpoint Did

- Reviewed the release commit log against the changelog and downstream docs.
- Refreshed current release docs and removed obsolete pre-0.7 public-facing wording.
- Preserved the ignored/local-only archive policy for `docs-archive/`.
- Confirmed the remaining release blocker is the operator-owned signing stage,
  not an agent-fixable lint or documentation issue.

## Next Session, Start Here

1. Confirm the operator has completed signed allowlist repair for
   `elspeth-2670a38693`.
2. Rerun or inspect Static analysis and aggregate CI on the current PR head.
3. Run the final live-judge verification path.
4. Update PR #86 and release provenance with the final CI/signing evidence.
