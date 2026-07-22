# ADR-024: Delivery Governance for Single-Maintainer Mode

**Date:** 2026-05-19
**Status:** Accepted
**Deciders:** ELSPETH maintainer
**Tags:** governance, cicd, release-management, branch-protection, elspeth-lints

## Context

ELSPETH started as a public open-source project maintained by one developer. It is now being built out as government-directed work while still having one assigned developer. That creates a governance tension:

- government-facing delivery needs explicit control evidence;
- the repository cannot honestly use two-person review while only one developer is assigned;
- self-approval would add ceremony without improving safety;
- automated controls are already load-bearing through CI, `elspeth-lints`, CodeQL, redaction gates, signed container images, and release smoke tests.

Without a recorded decision, a reviewer could misread `required_approving_review_count: 0` as an accidental waiver of review discipline. The real posture is different: ELSPETH is in a deliberate single-maintainer mode with compensating automated controls, and it has a defined step-up path for the moment a second developer is assigned.

## Decision

ELSPETH will operate in **single-maintainer mode** until a second developer receives regular write access or participates in release-critical delivery.

In single-maintainer mode:

- human approval count remains zero because self-review is not a meaningful control;
- default-branch and release-critical merges must be protected by mandatory automated gates;
- required evidence comes from CI, policy lints, CodeQL, redaction governance, artifact provenance, signatures, and smoke tests;
- release images must be tied to commits that have passed the required CI gate;
- advisory quality signals must be labelled as advisory rather than presented as enforced thresholds.

Every process, gate, and document must materially improve at least one of:

- reliability of code or tests;
- integrity of code, tests, runtime data, audit evidence, or documentation; or
- supportability of code, deployments, operations, or user workflows.

Plans, run sheets, test procedures, runbooks, and incident diagnostics pass
this test when they help build, verify, or operate the system. They are ordinary
working documents: update or delete them as the system changes. Do not sign or
seal plans, generate plan hash manifests or review-receipt sidecars, construct
approval chains, or require role handoffs merely to authenticate disposable
working documents. These controls simulate a multi-person organisation without
reducing product risk.

This exclusion does not apply to controls protecting actual system assets:
source commits, releases, images, exports, audit chains, runtime data, deployed
artifacts, and their admission evidence may still require hashes, signatures,
independent automated checks, or fail-closed gates. A gate that no longer
prevents a concrete failure should be removed. If removing a practice is a
marginal call or may discard a real safeguard, the tradeoff must be surfaced to
the maintainer before removal.

When a second developer is assigned, ELSPETH will step up to **two-maintainer mode** by enabling:

- one required approving review;
- stale-review dismissal on new commits;
- last-push approval protection;
- required conversation resolution;
- CODEOWNERS or an equivalent ownership map for security-sensitive paths;
- review requirements for release tags or release branches where GitHub supports them.

## Consequences

### Positive Consequences

- The current zero-reviewer setting is explainable as an honest staffing-mode decision, not an uncontrolled gap.
- Automated controls become more important and must be wired as required checks, not merely present as optional workflows.
- The project can move quickly while there is one maintainer without pretending to have a team process.
- The step-up path is already defined for government review and for future maintainers.

### Negative Consequences

- Single-maintainer mode still has concentration risk: one person can author and merge changes if automated checks pass.
- Some controls remain platform configuration rather than repository-tracked code, so periodic ruleset inspection is necessary.
- Review quality depends heavily on CI coverage, policy lints, and disciplined issue/ADR records until a second maintainer exists.

### Neutral Consequences

- Branch protection and repository rulesets are part of the delivery architecture.
- `elspeth-lints` and CodeQL are governance evidence, not just developer convenience tools.
- Mutation testing may remain advisory, but it must not be described as an enforced threshold until it actually fails builds.

## Alternatives Considered

### Require one approving review immediately

**Description:** Configure GitHub to require one human approval even while there is only one developer.

**Rejected because:** This would force self-approval or block delivery. Self-approval would be theatre and would teach reviewers to distrust the rest of the control set.

### Leave the posture implicit

**Description:** Keep the current practical setup and explain it verbally when asked.

**Rejected because:** Government-facing delivery needs durable evidence. The important distinction between "no review control" and "single-maintainer mode with compensating automated controls" should be discoverable without oral history.

### Use only manual release discipline

**Description:** Rely on the maintainer to remember which checks to run before release.

**Rejected because:** ELSPETH's safety model already favours mechanical enforcement over memory. Release and merge controls should fail closed wherever the platform can support it.

## Related Decisions

- ADR-023: Custom Python Static Analyzer for ELSPETH-Specific CI Invariants (the `elspeth-lints` Package)

## References

- [elspeth-lints rationale](../../elspeth-lints/rationale.md)
- [CI workflow](../../../.github/workflows/ci.yaml)
- [CodeQL workflow](../../../.github/workflows/codeql.yaml)
- [Build and push workflow](../../../.github/workflows/build-push.yaml)
- [Composer redaction gate](../../../.github/workflows/composer-redaction-gate.yml)

## Notes

This ADR does not claim ELSPETH has reached a mature multi-maintainer operating model. It records the current operating mode and the controls that keep it defensible until staffing catches up with the project risk profile.
