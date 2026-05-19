# `docs/release/` — Release Histories and Snapshots

**Last reviewed:** 2026-05-19 (RC-5.2)

This directory holds release-level documentation: cumulative progress reports,
velocity records, public-sector evaluation material, and visible historical
release snapshots.

The directory is intentionally small. **Current** documents reflect the latest
release (RC-5.2). **Historical** documents remain here only when they are useful
as visible release history. Superseded internal release truth docs were moved to
the [docs cleanout archive](../../docs-archive/2026-05-19-docs-cleanout/MANIFEST.md).
New readers should start with the current tier.

## Current (RC-5.2, May 2026)

| Document | What it answers | Intended reader |
|----------|-----------------|-----------------|
| [`executive-summary.md`](executive-summary.md) | What should a senior public-sector evaluator know before considering ELSPETH? | **Public-service executives, programme sponsors, assurance and risk staff.** Draft pending review, but kept visible because it was written for this audience. |
| [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md) | What has the project shipped between RC-1 and RC-5? Period-by-period, capability-by-capability, with cumulative commit counts. | **Engineering team and engineering leadership.** Detailed engineering content; not written for non-engineering stakeholders. |
| [`elspeth-velocity-rc1-to-rc5.md`](elspeth-velocity-rc1-to-rc5.md) | How much work was completed per day? Per-day commit volume across all 123 active days, with peak-day attribution. | **Engineering team and engineering leadership.** Per-commit cadence detail. |

The progress and velocity reports are designed as a **pair**: the progress doc
covers *what*, the velocity doc covers *when*. Each opens with a cross-reference
to the other.

## Assurance Narrative

The old `guarantees.md` was an RC-3-era hidden release document that looked like
current truth. Its durable content is now visible as
[`../contracts/assurance-contract.md`](../contracts/assurance-contract.md). The
stale release wrapper was deleted rather than kept here unread.

## Historical Snapshots

These documents captured ELSPETH at a particular RC and are no longer maintained.
Read them for historical context; do **not** treat their content as current.

| Document | Captured at | Stale as of | Notes |
|----------|-------------|-------------|-------|
| [`rc-3-release-notes.md`](rc-3-release-notes.md) | RC-3 release | Superseded by `CHANGELOG.md` | Detailed RC-3 release notes |

## Archived Elsewhere

These release documents were too stale or too hidden to remain in active docs.
They are preserved in the docs archive for provenance.

| Document | Captured at | Current replacement |
|----------|-------------|---------------------|
| [`feature-inventory.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/feature-inventory.md) | RC-3.3 — 1 March 2026 | Current progress report and `docs/reference/` |
| [`rc4-executive-brief.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc4-executive-brief.md) | RC-4.0 planning — 3 March 2026 | RC-5.2 executive summary, progress report, and current composer docs |
| [`rc-2-checkpoint-fix.md`](../../docs-archive/2026-05-19-docs-cleanout/docs/release/rc-2-checkpoint-fix.md) | RC-2 hot-fix note | `../runbooks/resume-failed-run.md` and Git history |
| `guarantees.md` | RC-3 assurance narrative | `../contracts/assurance-contract.md` |

## Canonical Sources of Truth

- **Per-release detailed changelog:** `/CHANGELOG.md` — the authoritative line-by-line release record that the documents in this directory summarise.
- **Live work tracking:** Filigree (`filigree session-context`) — what is in progress *now*; the progress report's "what's next" section pulls forward-looking items from here.
- **In-code architecture analyses:** `/docs/architecture/` and the layer-import graph emitted by `scripts/cicd/enforce_tier_model.py dump-edges`.

## When to add a new document here

- **Yes:** another comprehensive release summary at the next major RC; a new assurance-narrative snapshot if the assurance contract is restructured; a release brief if the team needs an executive-tier write-up for a particular RC.
- **No:** detailed per-feature design docs (those belong in `/docs/superpowers/specs/`); operational runbooks (`/docs/runbooks/`); ADRs (`/docs/adr/`).

When a new comprehensive summary supersedes an existing one (e.g. an "RC-1 to RC-6" report supersedes the current "RC-1 to RC-5" pair), update this README's *Current* table and move the superseded docs to the *Archived* table with the date they froze.
