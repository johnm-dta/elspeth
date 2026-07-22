# Permanent DAG Information Area Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `docs/architecture/dag/` as the permanent, iterative home for DAG completeness criteria, assessment practice, current status, and dated evidence.

**Architecture:** Separate evergreen guidance from immutable assessment snapshots. The root README presents the current verdict and update workflow; criteria define what “complete” means; the framework defines how to collect and score evidence; dated assessment directories retain the full provenance of each analysis.

**Tech Stack:** Markdown, Mermaid, repository-relative links, existing documentation validation tools.

---

## Task 1: Migrate the current assessment snapshot

**Files:**

- Move: `docs/arch-analysis-2026-07-15-1415/` to `docs/architecture/dag/assessments/2026-07-15-1415/`
- Reorganize: `docs/architecture/dag/assessments/2026-07-15-1415/temp/`

- [x] **Step 1: Move the full same-day assessment without dropping files**

Retain all numbered reports, evidence reports, validation output, and worker task provenance.

- [x] **Step 2: Replace the temporary directory name with durable categories**

Move evidence and validation reports to `evidence/`; move worker task briefs to `provenance/`.

- [x] **Step 3: Repair relative links**

Update links in the numbered reports so every evidence and cross-report link resolves from its new location.

## Task 2: Add the evergreen DAG completeness contract

**Files:**

- Create: `docs/architecture/dag/completeness-criteria.md`

- [x] **Step 1: Define the product boundary and completion rule**

Capture the full authoring-to-recovery lifecycle, mandatory dimensions, hard gates, and mandatory topology corpus from the approved current analysis.

- [x] **Step 2: Define explicit evidence standards**

Distinguish modeled, compiled, happy-path, production-supported, maintained, and unknown evidence without allowing averages to conceal hard-gate failures.

## Task 3: Add the repeatable assessment framework

**Files:**

- Create: `docs/architecture/dag/assessment-framework.md`

- [x] **Step 1: Define assessment inputs and workflow**

Require a fixed commit, fresh structural index, live tracker reconciliation, source/contracts, exact test commands, and explicit unknowns.

- [x] **Step 2: Add reusable scorecard and scenario templates**

Provide dimension, scenario, defect, and release-gate tables that future assessments can copy without inventing a new scoring method.

- [x] **Step 3: Define snapshot and current-status update rules**

Explain when to create a dated assessment, how to preserve prior findings, and how the hub points to the latest evidence.

## Task 4: Add the permanent navigation hub

**Files:**

- Create: `docs/architecture/dag/README.md`
- Modify: `docs/README.md`

- [x] **Step 1: Write the hub**

Lead with the current `2.4/5`, not-complete verdict, distinguish current from historical material, and link criteria, framework, latest assessment, evidence, and prior scheduler architecture.

- [x] **Step 2: Update the repository documentation index**

Replace the temporary analysis path with the permanent DAG information-area link while preserving the user’s existing index edit.

## Task 5: Verify the documentation set

**Files:**

- Verify: `docs/architecture/dag/**/*.md`
- Verify: `docs/README.md`

- [x] **Step 1: Check preservation**

Compare the pre- and post-migration file inventory and confirm that every source file has a durable destination.

- [x] **Step 2: Check links, placeholders, and whitespace**

Run repository Markdown checks when available, a local relative-link check, `rg -n 'TBD|TODO' docs/architecture/dag`, and `git diff --check`.

- [x] **Step 3: Review the final diff**

Confirm the change only establishes the permanent DAG area, migrates the approved current assessment, and updates navigation.
