# `docs/release/` — Release Histories and Snapshots

**Last reviewed:** 2026-07-14 (0.7.1)
**Audience:** Anyone navigating to release-level documentation
**Register:** Lightly technical / directory-index

This directory holds release-level documentation: cumulative progress reports,
velocity records, public-sector evaluation material, and the long-lived
assurance narrative.

The directory is intentionally small. **Current** documents reflect the release
currently being prepared (0.7.1). Superseded point-in-time release docs and frozen historical
snapshots are no longer part of the active public docs tree; use git history or
maintainer-local archives when historical provenance is needed. New readers
should start with the current tier below.

---

## Who should read what

If you only have time for one document, pick the row that matches your role.

| If you are… | Read first | Then |
|---|---|---|
| **An Australian public-service executive or assurance reader** evaluating the platform for pilot adoption | [`executive-summary.md`](executive-summary.md) | [`guarantees.md`](guarantees.md) for contractual claims |
| **Engineering leadership** scoping or sequencing work against the platform | [`platform-architecture.md`](platform-architecture.md) | Then `/CHANGELOG.md` for release-by-release implementation detail |
| **An engineer joining the project** | [`platform-architecture.md`](platform-architecture.md) | Then [`guarantees.md`](guarantees.md) for the contracts the code must uphold |
| **An auditor or integrator** verifying assurance claims | [`guarantees.md`](guarantees.md) (the assurance surface) | Then `/CHANGELOG.md` (RC-3+) for the line-by-line release record |
| **Anyone investigating a specific historical decision or release** | The relevant commit in git history or a maintainer-local archive | The current release documents for subsequent context |

---

## Canonical sources of truth

Before diving into the documents below, know which sources they aggregate. **If a document in this directory contradicts a canonical source, the canonical source wins.**

- **Per-release detailed changelogs:** [`/CHANGELOG.md`](../../CHANGELOG.md) (RC-3+) — line-by-line release records.
- **Architectural decisions:** [`/docs/architecture/adr/`](../architecture/adr/) — the binding ADRs.
- **In-code architecture analyses:** [`/docs/architecture/`](../architecture/) and the layer-import graph emitted by `elspeth-lints dump-edges`.

---

## Current documents (0.7.1, July 2026)

| Document | What it answers | Intended reader |
|----------|-----------------|-----------------|
| [`executive-summary.md`](executive-summary.md) *(DRAFT — awaiting operator sign-off)* | Capability and assurance summary; what the platform does, what it does not yet guarantee, what an evaluator should consider next. | **Public-service executives, programme sponsors, assurance and risk staff.** Australian public-service / institutional register. |
| [`composer-guide.md`](composer-guide.md) | What the Composer can do, how guided/freeform authoring works, what readiness checks mean, and how a user completes or recovers a composition. | **Evaluators, programme teams, operators, and technical reviewers.** Public-facing / lightly technical. |
| [`platform-architecture.md`](platform-architecture.md) | Current runtime surfaces, trust boundaries, audit-first behaviour, configuration validation, external-system boundaries, and adopter responsibilities. | **Evaluators, technical leaders, architects, and assurance reviewers.** Public-facing / technical. |
| [`assessment-mapping.md`](assessment-mapping.md) | How the current evidence set maps to likely public-sector evaluation questions, without claiming formal conformance. | **Assurance, risk, security, delivery governance, and agency evaluation teams.** Evidence map. |
| [`sink-effect-recovery.md`](../runbooks/sink-effect-recovery.md) | The durable sink-effect recovery protocol, target-ledger requirements, and operator decisions after interrupted publication. | **Operators, plugin maintainers, and database owners.** Recovery runbook. |

The executive summary is the single executive-tier entry point. Engineering
readers should use the platform architecture and root changelog for current
implementation context.

## Assurance narrative (long-lived; refreshed per release)

| Document | What it answers | Status |
|----------|-----------------|--------|
| [`guarantees.md`](guarantees.md) | Audit, lineage, and trust-model guarantees ELSPETH makes. | **Layered assurance appendix.** §1–§10 preserve the original RC-3 contract language; §11–§14 add RC-5.2 authentication, secret-reference, multi-user-session, and composer-authoring guarantees; §15 adds durable token scheduling guarantees. The 0.7.1 refresh keeps this as the current assurance baseline. |

## Release PDF pack

The [`pdf/`](pdf/) pipeline contains the legacy RC-5.2 PDF/UA-1 release pack
sources. It remains useful as a build/reference pipeline, but it has not been
rebuilt as a 0.7.1 public release pack. Treat the Markdown documents in this
directory, the root changelog, and root architecture overview as the current
0.7.1 release sources until a new PDF pack is generated.

## Archived snapshots

Older release-cut and point-in-time documents are preserved through git history
and optional maintainer-local archives. They are no longer maintained; read them
for historical context only, not as current release guidance.

---

## When to add a new document here

- **Yes:** another comprehensive release summary at the next major RC; a refresh of the assurance narrative when `guarantees.md` is restructured; a release brief if the team needs an executive-tier write-up for a particular RC.
- **No:** detailed per-feature design docs after implementation; operational
  runbooks (`/docs/runbooks/`); ADRs (`/docs/architecture/adr/`).

When a new comprehensive summary supersedes an existing one, update this
README's *Current documents* table and remove the superseded document from
active docs. Maintainers may preserve it in a local ignored archive with the
date it froze. Apply the archive banner template (below) before archival.

## Archive banner template

Apply this banner verbatim at the top of any document before it is moved out of
active docs. It defends deep-link readers in git history or maintainer-local
archives against picking up stale content without the framing:

```markdown
> **ARCHIVED — <doc-class> captured at <date> (<release>).**
> This document records <topic> as it stood at <release>.
> <List the major intervening changes that make this document stale.>
>
> **For the current state, see** <link to the canonical current doc>.
```

Doc-class values currently in use: `Release notes`, `Post-mortem`, `Planning brief`, `Feature-inventory snapshot`. Add new classes only when an existing class genuinely does not fit.

In addition to the banner, every archived doc carries a metadata block under its title naming **Audience** and **Register** so that a deep-link reader sees the audience contract without needing to come back to this index.
