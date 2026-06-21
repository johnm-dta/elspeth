# `docs/release/` — Release Histories and Snapshots

**Last reviewed:** 2026-06-22 (0.6.0)
**Audience:** Anyone navigating to release-level documentation
**Register:** Lightly technical / directory-index

This directory holds release-level documentation: cumulative progress reports,
velocity records, public-sector evaluation material, and the long-lived
assurance narrative.

The directory is intentionally small. **Current** documents reflect the latest
release (0.6.0). Superseded point-in-time release docs and frozen historical
snapshots were relocated to the
[2026-05-19 docs cleanout archive](../../docs-archive/2026-05-19-docs-cleanout/MANIFEST.md);
that archive's MANIFEST is the relocation map. New readers should start with the
current tier below.

---

## Who should read what

If you only have time for one document, pick the row that matches your role.

| If you are… | Read first | Then |
|---|---|---|
| **An Australian public-service executive or assurance reader** evaluating the platform for pilot adoption | [`executive-summary.md`](executive-summary.md) | [`guarantees.md`](guarantees.md) for contractual claims |
| **Engineering leadership** scoping or sequencing work against the platform | [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md) (the cumulative output snapshot at the foot) | Then sample the *Velocity by Phase* table in [`elspeth-velocity-rc1-to-rc5.md`](elspeth-velocity-rc1-to-rc5.md) |
| **An engineer joining the project** | [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md) in full, by Period | Then [`guarantees.md`](guarantees.md) for the contracts the code must uphold |
| **An auditor or integrator** verifying assurance claims | [`guarantees.md`](guarantees.md) (the assurance surface) | Then `/CHANGELOG.md` (RC-3+) for the line-by-line release record |
| **Anyone investigating a specific historical decision or release** | The relevant document in the [2026-05-19 docs archive](../../docs-archive/2026-05-19-docs-cleanout/docs/release/) | The current progress report for any subsequent context |

---

## Canonical sources of truth

Before diving into the documents below, know which sources they aggregate. **If a document in this directory contradicts a canonical source, the canonical source wins.**

- **Per-release detailed changelogs:** [`/CHANGELOG.md`](../../CHANGELOG.md) (RC-3+), and the archived [`CHANGELOG-RC2.md`](../../docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC2.md) / [`CHANGELOG-RC1.md`](../../docs-archive/2026-05-19-docs-cleanout/CHANGELOG-RC1.md) — line-by-line release records.
- **Architectural decisions:** [`/docs/architecture/adr/`](../architecture/adr/) — the binding ADRs.
- **In-code architecture analyses:** [`/docs/architecture/`](../architecture/) and the layer-import graph emitted by `elspeth-lints dump-edges`.

---

## Current documents (0.6.0, June 2026)

| Document | What it answers | Intended reader |
|----------|-----------------|-----------------|
| [`executive-summary.md`](executive-summary.md) *(DRAFT — awaiting operator sign-off)* | Capability and assurance summary; what the platform does, what it does not yet guarantee, what an evaluator should consider next. | **Public-service executives, programme sponsors, assurance and risk staff.** Australian public-service / institutional register. |
| [`composer-guide.md`](composer-guide.md) | What the Composer can do, how guided/freeform authoring works, what readiness checks mean, and how a user completes or recovers a composition. | **Evaluators, programme teams, operators, and technical reviewers.** Public-facing / lightly technical. |
| [`platform-architecture.md`](platform-architecture.md) | Current runtime surfaces, trust boundaries, audit-first behaviour, configuration validation, external-system boundaries, and adopter responsibilities. | **Evaluators, technical leaders, architects, and assurance reviewers.** Public-facing / technical. |
| [`assessment-mapping.md`](assessment-mapping.md) | How the RC-5.3 evidence set maps to likely public-sector evaluation questions, without claiming formal conformance. | **Assurance, risk, security, delivery governance, and agency evaluation teams.** Evidence map. |
| [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md) | What has the project shipped between RC-1 and RC-5? Period-by-period, capability-by-capability, with cumulative commit counts. | **Engineering team and engineering leadership.** Detailed engineering content; not written for non-engineering stakeholders. |
| [`elspeth-velocity-rc1-to-rc5.md`](elspeth-velocity-rc1-to-rc5.md) | How much work was completed per day? Per-day commit volume across all 123 active days, with peak-day attribution. | **Engineering team and engineering leadership.** Per-commit cadence detail; commit count is a tempo signal, not a value signal. |

The two engineering documents are designed as a **pair**: the progress doc covers *what*, the velocity doc covers *when*. Each opens with a cross-reference to the other. The executive summary is the single executive-tier entry point and references both engineering documents as back-stops.

## Assurance narrative (long-lived; refreshed per release)

| Document | What it answers | Status |
|----------|-----------------|--------|
| [`guarantees.md`](guarantees.md) | Audit, lineage, and trust-model guarantees ELSPETH makes. | **Layered assurance appendix.** §1–§10 preserve the original RC-3 contract language; §11–§14 add RC-5.2 authentication, secret-reference, multi-user-session, and composer-authoring guarantees. §7.2 has been amended to reflect that the "ELSPETH is not multi-user" disclaimer is no longer accurate. |

## Release PDF pack

The [`pdf/`](pdf/) pipeline builds PDF/UA-1 reading copies for release
briefing, engineering evidence, architecture, Composer, assurance, and
data-trust material. The generated PDFs live under `pdf/out/` and are
regeneratable build artefacts; the Typst sources and build pipeline are
the maintained release assets.

Current outputs:

- `elspeth-executive-summary.pdf`
- `elspeth-progress-rc1-to-rc5.pdf`
- `elspeth-velocity-rc1-to-rc5.pdf`
- `elspeth-architecture.pdf`
- `elspeth-composer.pdf`
- `elspeth-guarantees.pdf`
- `elspeth-data-trust.pdf`

## Archived snapshots

Older release-cut and point-in-time documents are preserved in the dated 2026-05-19 docs archive. They are no longer maintained; read for historical context, but do **not** treat their content as current.

| Document | Captured at | Stale as of | Notes |
|----------|-------------|-------------|-------|
| [`feature-inventory.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/feature-inventory.md) | RC-3.3 — 1 March 2026 | RC-3.4 onwards | Comprehensive plugin / capability inventory; superseded by the *Cumulative Output Snapshot* in the current progress report |
| [`rc4-executive-brief.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc4-executive-brief.md) | RC-4.0 planning — 3 March 2026 | RC-5 cut (3 April 2026) | Forward-looking work-package brief for the planned RC-4; what was scoped here ultimately shipped as the RC-5 Web UX Composer. Includes a *What actually shipped* appendix mapping each planned feature to its RC-5 outcome. |
| [`rc-3-release-notes.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc-3-release-notes.md) | RC-3 release | Superseded by `/CHANGELOG.md` | Detailed RC-3 release notes |
| [`rc-2-checkpoint-fix-postmortem.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc-2-checkpoint-fix-postmortem.md) | RC-2 hot-fix post-mortem | Superseded by `CHANGELOG-RC2.md` | Single-defect post-mortem (P1-2026-01-21) — preserved for the Lessons Learned and Prevention sections |
| [`rc-2-checkpoint-fix.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc-2-checkpoint-fix.md) | RC-2 hot-fix release note | Migration content folded into [`../runbooks/resume-failed-run.md`](../runbooks/resume-failed-run.md) | Original release-note framing kept in the archive for provenance |
| [`rc2-checklist.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc2-checklist.md) | RC-2 release gate | Obsolete | Original release-gate checklist |
| [`rc3-checklist.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc3-checklist.md) | RC-3 release gate | Obsolete | Original release-gate checklist |

---

## When to add a new document here

- **Yes:** another comprehensive release summary at the next major RC; a refresh of the assurance narrative when `guarantees.md` is restructured; a release brief if the team needs an executive-tier write-up for a particular RC.
- **No:** detailed per-feature design docs (those belong in `/docs/superpowers/specs/`); operational runbooks (`/docs/runbooks/`); ADRs (`/docs/architecture/adr/`).

When a new comprehensive summary supersedes an existing one (e.g. an "RC-1 to RC-6" report supersedes the current "RC-1 to RC-5" pair), update this README's *Current documents* table and relocate the superseded docs into a new dated subdirectory under `docs-archive/` with the date they froze. Apply the archive banner template (below) on the moved document.

## Archive banner template

Apply this banner verbatim at the top of any document moved to `docs-archive/`. It defends deep-link readers against picking up stale content without the framing:

```markdown
> **ARCHIVED — <doc-class> captured at <date> (<release>).**
> This document records <topic> as it stood at <release>.
> <List the major intervening changes that make this document stale.>
>
> **For the current state, see** <link to the canonical current doc>.
```

Doc-class values currently in use: `Release notes`, `Post-mortem`, `Planning brief`, `Feature-inventory snapshot`. Add new classes only when an existing class genuinely does not fit.

In addition to the banner, every archived doc carries a metadata block under its title naming **Audience** and **Register** so that a deep-link reader sees the audience contract without needing to come back to this index.
