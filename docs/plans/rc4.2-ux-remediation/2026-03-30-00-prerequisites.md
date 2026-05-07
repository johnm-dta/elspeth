# RC4.2 UX Remediation — Phase 0 Prerequisites

Date: 2026-03-30
Status: Draft
Parent: `docs/plans/rc4.2-ux-remediation/2026-03-30-rc4.2-ux-implementation-plan.md`

---

## 1. Purpose

This document captures the implementation prerequisites that must land before
the main RC4.2 remediation subplans begin in earnest.

These are not product features. They are enabling work that prevents the first
implementation agents from immediately tripping over missing infrastructure.

---

## 2. Required Before Implementation

### PREREQ-01: Frontend test harness

Needed before:

- [2026-03-30-01-chat-polish-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-01-chat-polish-subplan.md)

Why:

- sub-plan `01` is primarily frontend state-machine work
- the repo currently does not have a frontend unit/component test runner
- the plan explicitly expects store/component tests for send-state and scroll
  behavior

Minimum deliverables:

- install `vitest`
- install `@testing-library/react`
- add the minimal test configuration needed to run frontend unit/component
  tests
- define a repeatable Zustand store test pattern for isolated store-state tests

Success criteria:

- frontend tests can run locally in one command
- a minimal smoke test passes against the web frontend test environment

---

### PREREQ-02: Sessions DB migration infrastructure

Needed before:

- [2026-03-30-02-blob-manager-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-02-blob-manager-subplan.md)
- [2026-03-30-03-secret-references-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-03-secret-references-subplan.md)
- [2026-03-30-04-fork-from-message-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-04-fork-from-message-subplan.md)

Why:

- these subplans add new tables and columns to `sessions.db`
- relying on `create_all()` drift is not acceptable once schema evolution
  begins across multiple waves
- the plans already assume a real migration path

Minimum deliverables:

- bootstrap or extend migration tooling for `sessions.db`
- make the migration path explicit for:
  - new tables
  - new columns
  - backfills where required
- document how schema changes are applied in development and test

Success criteria:

- a migration can add a table/column to `sessions.db` deterministically
- test and local-dev environments can apply migrations cleanly

---

## 3. Recommended Execution Order

1. Complete `PREREQ-01` frontend test harness.
2. Complete `PREREQ-02` sessions DB migration infrastructure.
3. Start the numbered RC4.2 remediation subplans in sequence.

Practical implementation order after prerequisites:

1. [2026-03-30-01-chat-polish-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-01-chat-polish-subplan.md)
2. [2026-03-30-02-blob-manager-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-02-blob-manager-subplan.md)
3. [2026-03-30-03-secret-references-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-03-secret-references-subplan.md)
4. [2026-03-30-04-fork-from-message-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-04-fork-from-message-subplan.md)
5. [2026-03-30-05-validation-and-validation-ux-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-05-validation-and-validation-ux-subplan.md)
6. [2026-03-30-06-composer-api-enhancements-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-06-composer-api-enhancements-subplan.md)
7. [2026-03-30-07-inspector-ux-subplan.md](/home/john/elspeth/docs/plans/rc4.2-ux-remediation/2026-03-30-07-inspector-ux-subplan.md)

---

## 4. Notes

- These prerequisites are intentionally small and enabling in nature.
- They should be treated as blockers, not optional cleanup.
- Once these are done, the rest of the remediation sequence can proceed with
  much less implementation friction.
