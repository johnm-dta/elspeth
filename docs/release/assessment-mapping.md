# RC-5.2 Public-Sector Assessment Mapping

**Document date:** 20 May 2026
**Release covered:** RC-5.2
**Audience:** Assurance, risk, security, delivery governance, and agency evaluation teams
**Register:** Australian public-sector evaluation / evidence map
**Status:** Current evidence map; not a conformance statement

## Scope

This document maps the current ELSPETH RC-5.2 evidence set against the public
sector touchpoints an evaluator is likely to raise. It is deliberately not a
certification, IRAP report, Digital Service Standard assessment, Essential Eight
maturity assessment, PSPF statement, AGDS assessment, or WCAG conformance claim.

Canonical project evidence remains:

- `CHANGELOG.md` for release-by-release detail;
- `docs/release/executive-summary.md` for executive posture and residual risk;
- `docs/release/composer-guide.md` for the current web authoring surface;
- `docs/release/platform-architecture.md` for the current architecture and
  trust-boundary overview;
- `docs/release/guarantees.md` for audit and lineage guarantees;
- `docs/runbooks/` for operational procedures;
- `.github/workflows/` and `docs/runbooks/ci-branch-protection.md` for CI and
  branch-protection posture.

## External Reference Points

Evaluators are likely to ask how the evidence relates to public-sector
reference points such as the Digital Service Standard, ASD ISM software
development guidance, Essential Eight maturity, PSPF responsibilities, AGDS,
WCAG, IRAP assessment, and authority-to-operate boundaries.

ELSPETH does not claim formal conformance to those frameworks in RC-5.2. This
document identifies the evidence already present and the gaps an adopting agency
would need to close.

## Current Evidence Map

| Area | RC-5.2 evidence present | Gap or caveat |
|---|---|---|
| Clear intent and audience | README, executive summary, progress report, and guarantees document position ELSPETH as a high-assurance pipeline substrate for auditable workflows. | No formal Digital Service Standard checklist has been completed. |
| User and service understanding | The Composer guide, platform architecture overview, UX redesign docs, evidence reports, Playwright tests, and executive summary describe target operator and knowledge-worker audiences. | No independent user-research dossier or AGDS assessment is included in the release bundle. |
| Accessibility and inclusive design | Executive summary records skip-to-content navigation, reduced-motion support, and screen-reader-accessible status indicators. CI includes frontend unit/typecheck and Playwright lanes. | No formal WCAG or AGDS conformance assessment has been completed. |
| Security governance | `SECURITY.md`, `GOVERNANCE.md`, branch-protection runbook, CodeQL, elspeth-lints, Dependabot, pip-audit, license audit, redaction gate, and CI success aggregation provide a governance surface. | Security disclosure still needs a durable private public channel before broad announcement. |
| Authoritative source and issue tracking | GitHub repository, branch protection, signed/pinned workflow actions, and issue tracking are present. | Branch-protection and release-gate evidence should be refreshed against the final release commit before publication. |
| Software artifacts | Build workflow is configured for multi-platform images with SBOM, provenance, and cosign signing. | Final RC-5.2 release artifacts, immutable digest, public verification commands, and evidence bundle must be attached to the release record before claiming publication. |
| SAST/SCA and dependency controls | CodeQL, pip-audit, license audit, Dependabot, and custom elspeth-lints are configured. | DAST and secret-scanning evidence are not yet packaged into the release evidence bundle. |
| Audit and logging | Landscape audit is the canonical evidentiary record. Logging and telemetry policy is documented; audit-first writes are central product behaviour. | Independent assessment outside the scoped interim ATO has not been completed. |
| Authentication and identity | RC-5.2 supports local, OIDC, and Microsoft Entra authentication with principal recording claims in guarantees. | Organisation-level authorisation policy remains a deployer responsibility. |
| Secret handling | Secrets are referenced by name, resolved at runtime, and fingerprinted for audit. | Secret scanning for repository commits should be evidenced before public release. |
| AI and prompt-injection concerns | Composer redaction, tool-call persistence, LLM audit records, guided-mode constraints, and interpretation review are documented. | A formal AI-specific threat model mapped to current agency controls has not been packaged as a release artifact. |
| Web application controls | FastAPI/React architecture, auth providers, sessions, secure cookie expectations, SSRF controls for web scraping, and runbooks exist. | No independent penetration test or OWASP ASVS assessment has been completed. |
| Operations and recovery | Runbooks cover incident response, backup/recovery, database maintenance, resume, routing investigation, Key Vault, Ansible deployment, and staging DB recreation. | Manual database recreation remains an operational risk for some upgrades. |
| Continuity | ADRs, contracts, runbooks, CI gates, and tests mitigate knowledge concentration. | Maintainer continuity remains material and must stay visible in public release language. |

## Release-Readiness Interpretation

ELSPETH has unusually strong internal evidence for auditability and release
discipline, but public-sector readiness depends on not overstating that
evidence. The honest current posture is:

- suitable for evaluation and scoped pilot discussion;
- not a whole-platform certified government service;
- not yet accompanied by a formal DSS, ISM, Essential Eight, PSPF, AGDS, WCAG,
  IRAP, or penetration-test assessment;
- stronger when shipped with a release evidence bundle that includes CI,
  security, supply-chain, and residual-risk artifacts.

## Required Before Public Announcement

- Record final release approval in `executive-summary.md`.
- Publish or enable a durable private vulnerability reporting path.
- Refresh branch-protection evidence against the final release commit.
- Publish or explicitly mark unavailable each expected supply-chain artifact.
- Attach final CI, CodeQL, dependency/license, Playwright, and branch-protection
  evidence to the release record.
- Decide whether a formal DSS/ISM/Essential Eight/PSPF/AGDS/WCAG mapping is in
  scope for RC-5.2 or explicitly deferred.
