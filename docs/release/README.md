# `docs/release/` — Release Histories and Snapshots

**Last reviewed:** 2026-05-19 (RC-5.2)

This directory holds release-level documentation: cumulative progress reports, velocity records, point-in-time feature snapshots, planning briefs, and assurance narratives.

The directory is two-tiered. **Current** documents reflect the latest release (RC-5.2); **archived** documents capture earlier RC states and are retained for historical reference. New readers should start with the current tier.

## Current (RC-5.2, May 2026)

| Document | What it answers | Intended reader |
|----------|-----------------|-----------------|
| [`elspeth-progress-rc1-to-rc5.md`](elspeth-progress-rc1-to-rc5.md) | What has the project shipped between RC-1 and RC-5? Period-by-period, capability-by-capability, with cumulative commit counts. | **Engineering team and engineering leadership.** Detailed engineering content; not written for non-engineering stakeholders. |
| [`elspeth-velocity-rc1-to-rc5.md`](elspeth-velocity-rc1-to-rc5.md) | How much work was completed per day? Per-day commit volume across all 123 active days, with peak-day attribution. | **Engineering team and engineering leadership.** Per-commit cadence detail. |

These two are designed as a **pair**: the progress doc covers *what*, the velocity doc covers *when*. Each opens with a cross-reference to the other.

## Assurance Narrative (long-lived; refreshed per release)

| Document | What it answers | Status |
|----------|-----------------|--------|
| [`guarantees.md`](guarantees.md) | Audit, lineage, and trust-model guarantees ELSPETH makes | **STALE — RC-3 vintage (3 March 2026)**; not yet refreshed for RC-4 / RC-5 capability additions (auth, redaction, multi-user). A refresh to RC-5.2 is on the roadmap. Read with the progress report alongside until refreshed. |

## Archived Snapshots

These documents captured ELSPETH at a particular RC and are no longer maintained. Read them for historical context; do **not** treat their content as current.

| Document | Captured at | Stale as of | Notes |
|----------|-------------|-------------|-------|
| [`feature-inventory.md`](feature-inventory.md) | RC-3.3 — 1 March 2026 | RC-3.4 onwards | Comprehensive plugin / capability inventory; superseded by the *Cumulative Output Snapshot* in the current progress report |
| [`rc4-executive-brief.md`](rc4-executive-brief.md) | RC-4.0 planning — 3 March 2026 | RC-5 cut (3 April 2026) | Forward-looking work-package brief for the planned RC-4; what was scoped here ultimately shipped as the RC-5 Web UX Composer |
| [`rc-3-release-notes.md`](rc-3-release-notes.md) | RC-3 release | Superseded by `CHANGELOG.md` | Detailed RC-3 release notes |
| [`rc-2-checkpoint-fix.md`](rc-2-checkpoint-fix.md) | RC-2 hot-fix note | Superseded by `CHANGELOG-RC2.md` | Single-issue checkpoint-fix write-up |

## Canonical Sources of Truth

- **Per-release detailed changelogs:** `/CHANGELOG.md` (RC-3+), `/CHANGELOG-RC2.md`, `/CHANGELOG-RC1.md` — these are the authoritative line-by-line release records that the documents in this directory summarise.
- **Live work tracking:** Filigree (`filigree session-context`) — what is in progress *now*; the progress report's "what's next" section pulls forward-looking items from here.
- **In-code architecture analyses:** `/docs/architecture/` and the layer-import graph emitted by `scripts/cicd/enforce_tier_model.py dump-edges`.

## When to add a new document here

- **Yes:** another comprehensive release summary at the next major RC; a new assurance-narrative snapshot if `guarantees.md` is restructured; a release brief if the team needs an executive-tier write-up for a particular RC.
- **No:** detailed per-feature design docs (those belong in `/docs/superpowers/specs/`); operational runbooks (`/docs/runbooks/`); ADRs (`/docs/adr/`).

When a new comprehensive summary supersedes an existing one (e.g. an "RC-1 to RC-6" report supersedes the current "RC-1 to RC-5" pair), update this README's *Current* table and move the superseded docs to the *Archived* table with the date they froze.
