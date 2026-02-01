# Documentation Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate documentation debt accumulated during RC-1 bug hunt - consolidate overlapping folders, delete obsolete archives, fix content duplication, and establish clear navigation.

**Architecture:** Three-phase cleanup - structural consolidation first (move/delete folders), then content deduplication (merge repeated sections), then quality improvements (add missing metadata, fix broken references).

**Constraints:**
- **NO BASH for file operations** - Use Glob, Grep, Read tools. Only use Bash for git commands.
- All changes must preserve git history (use `git mv` not raw moves)
- Verify each deletion target is truly obsolete before removing

---

## Phase 1: Structural Consolidation

### Task 1.1: Archive the Dated Architecture Analysis Folder

**Goal:** Move `arch-analysis-2026-01-27-2132/` out of the main docs tree into archive.

**Files:**
- Move: `docs/arch-analysis-2026-01-27-2132/` → `docs/archive/2026-01-27-arch-analysis/`
- Delete: `docs/arch-analysis-2026-01-27-2132/temp/` (temporary working files)

**Step 1: Verify the folder exists and check contents**

Use Glob to list: `docs/arch-analysis-2026-01-27-2132/**/*.md`
Expected: 7+ markdown files (coordination, discovery, diagrams, final-report, quality, debt, handover)

**Step 2: Check if archive directory exists**

Use Glob to check: `docs/archive/*`
If `docs/archive/` doesn't exist, it will be created by git mv.

**Step 3: Move the folder using git**

```bash
git mv docs/arch-analysis-2026-01-27-2132 docs/archive/2026-01-27-arch-analysis
```

**Step 4: Delete the temp subfolder**

```bash
git rm -rf docs/archive/2026-01-27-arch-analysis/temp
```

**Step 5: Verify the move**

Use Glob: `docs/archive/2026-01-27-arch-analysis/**/*.md`
Expected: 7 files present, no `temp/` directory

**Step 6: Commit**

```bash
git add -A && git commit -m "docs: archive dated architecture analysis folder

Move arch-analysis-2026-01-27-2132 to archive/ and delete temp/ working files.
This was a one-time analysis session, not ongoing documentation.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.2: Move Misplaced Analysis Files to Quality-Audit

**Goal:** The `docs/analysis/` folder contains mutation testing content that belongs in `docs/quality-audit/`.

**Files:**
- Move: `docs/analysis/*.md` → `docs/quality-audit/`
- Delete: `docs/analysis/` (empty after move)

**Step 1: List files to move**

Use Glob: `docs/analysis/*.md`
Expected files:
- `TEST_SUITE_ANALYSIS_2026-01-22.md`
- `MUTATION_TESTING_SUMMARY_2026-01-25.md`
- `MUTATION_GAPS_CHECKLIST.md`
- `TEST_SUITE_ACTION_PLAN.md`
- `HOW_TO_FIX_SURVIVORS.md`

**Step 2: Check for conflicts in destination**

Use Glob: `docs/quality-audit/*.md`
Verify none of the source filenames already exist in destination.

**Step 3: Move each file**

```bash
git mv docs/analysis/TEST_SUITE_ANALYSIS_2026-01-22.md docs/quality-audit/
git mv docs/analysis/MUTATION_TESTING_SUMMARY_2026-01-25.md docs/quality-audit/
git mv docs/analysis/MUTATION_GAPS_CHECKLIST.md docs/quality-audit/
git mv docs/analysis/TEST_SUITE_ACTION_PLAN.md docs/quality-audit/
git mv docs/analysis/HOW_TO_FIX_SURVIVORS.md docs/quality-audit/
```

**Step 4: Remove empty analysis directory**

```bash
rmdir docs/analysis
```

If rmdir fails (directory not empty), use Glob to find remaining files and handle them.

**Step 5: Verify the move**

Use Glob: `docs/quality-audit/MUTATION*.md`
Expected: At least 2 files (MUTATION_TESTING_SUMMARY, MUTATION_GAPS_CHECKLIST)

**Step 6: Commit**

```bash
git add -A && git commit -m "docs: move mutation testing analysis to quality-audit

These files are test quality analysis, not architecture documentation.
Consolidating all quality/testing analysis in one location.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.3: Consolidate Token Lifecycle into Design Subsystems

**Goal:** Move `docs/architecture/token-lifecycle.md` into `docs/design/subsystems/` and delete the now-empty `architecture/` folder.

**Files:**
- Move: `docs/architecture/token-lifecycle.md` → `docs/design/subsystems/06-token-lifecycle.md`
- Delete: `docs/architecture/` (empty after move)

**Step 1: Verify source file exists**

Use Read: `docs/architecture/token-lifecycle.md` (first 50 lines)
Expected: Token lifecycle documentation with clear structure.

**Step 2: Check destination folder structure**

Use Glob: `docs/design/subsystems/*.md`
Expected: `00-overview.md` and possibly others. Note the numbering convention.

**Step 3: Move with rename**

```bash
git mv docs/architecture/token-lifecycle.md docs/design/subsystems/06-token-lifecycle.md
```

**Step 4: Remove empty architecture directory**

```bash
rmdir docs/architecture
```

**Step 5: Update cross-references in subsystems overview**

Use Read: `docs/design/subsystems/00-overview.md`
Search for references to `architecture/token-lifecycle.md` or token lifecycle section.
If found, update the reference to `06-token-lifecycle.md`.

Use Edit to fix any broken references found.

**Step 6: Commit**

```bash
git add -A && git commit -m "docs: consolidate token-lifecycle into design/subsystems

Move from standalone architecture/ folder into subsystems/ hierarchy.
This eliminates the confusing architecture/ vs design/ folder split.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.4: Delete Obsolete Bug Archives

**Goal:** Remove bug archive folders that contain only obsolete automation artifacts.

**Files to delete:**
- `docs/bugs/archive/generated-2026-01-22/` (456KB of one-off scan artifacts)
- `docs/bugs/archive/process-2026-01-22/` (automation logs)
- `docs/bugs/archive/scripts-obsolete-2026-01-24/` (explicitly marked obsolete)

**Files to KEEP:**
- `docs/bugs/archive/reports-2026-01-24/` (contains valuable VERIFICATION-REPORT)
- `docs/bugs/archive/resolved-2026-01-27/` (recent resolutions, review before deleting)

**Step 1: Verify what's being deleted**

Use Glob: `docs/bugs/archive/generated-2026-01-22/**/*.md`
Expected: ~40 files of source code analysis (recorder.py.md, canonical.py.md, etc.)

Use Read: `docs/bugs/archive/generated-2026-01-22/CATALOG.md` (first 30 lines)
Confirm this is automated scan output, not manually curated content.

**Step 2: Verify scripts-obsolete is safe to delete**

Use Glob: `docs/bugs/archive/scripts-obsolete-2026-01-24/*`
Use Read: `docs/bugs/archive/scripts-obsolete-2026-01-24/README.md`
Confirm it's marked obsolete.

**Step 3: Delete the folders**

```bash
git rm -rf docs/bugs/archive/generated-2026-01-22
git rm -rf docs/bugs/archive/process-2026-01-22
git rm -rf docs/bugs/archive/scripts-obsolete-2026-01-24
```

**Step 4: Verify remaining archive structure**

Use Glob: `docs/bugs/archive/*`
Expected: `reports-2026-01-24/` and `resolved-2026-01-27/` remain.

**Step 5: Commit**

```bash
git add -A && git commit -m "docs: delete obsolete bug archive artifacts

Remove:
- generated-2026-01-22/ (one-off automated scan output)
- process-2026-01-22/ (automation logs)
- scripts-obsolete-2026-01-24/ (explicitly marked obsolete)

Retain reports-2026-01-24/ for audit trail value.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.5: Clean Up Generated Bug Duplicates

**Goal:** Delete the 12 documented duplicates from `docs/bugs/generated/` that are already tracked in `docs/bugs/open/`.

**Files:**
- Reference: `docs/bugs/archive/DUPLICATES.md` (if exists) or identify duplicates manually
- Delete: Duplicate files in `docs/bugs/generated/`

**Step 1: Find the duplicates list**

Use Grep: Search for "duplicate" in `docs/bugs/archive/**/*.md`
Or use Glob: `docs/bugs/archive/**/DUPLICATES.md`

If DUPLICATES.md exists, Read it to get the list.

**Step 2: If no DUPLICATES.md, identify duplicates manually**

Use Glob: `docs/bugs/generated/**/*.md`
Use Glob: `docs/bugs/open/**/*.md`

Compare filenames - look for matching bug IDs or descriptions.

**Step 3: For each confirmed duplicate, delete from generated/**

Example (adjust based on actual duplicates found):
```bash
git rm docs/bugs/generated/by-priority/P1/duplicate_gate_names.md
git rm docs/bugs/generated/by-priority/P1/export_status_masking.md
# ... etc for each duplicate
```

**Step 4: Verify generated/ is cleaner**

Use Glob: `docs/bugs/generated/**/*.md | wc -l`
Expected: Fewer files than before (was 34, should be ~22 after removing duplicates)

**Step 5: Commit**

```bash
git add -A && git commit -m "docs: remove duplicate bugs from generated/

These bugs are already tracked in open/ with more detail.
Keeping only unique generated bugs that need triage.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 1.6: Archive Misc Azure Performance Docs

**Goal:** Move the Azure-specific performance analysis from `docs/misc/` to a structured archive location.

**Files:**
- Move: `docs/misc/AZURE_*.md` → `docs/archive/2026-01-azure-performance/`
- Move: `docs/misc/RETRYABLE_*.md` → `docs/archive/2026-01-azure-performance/`
- Move: `docs/misc/PROFILING_*.md` → `docs/archive/2026-01-azure-performance/`
- Move: `docs/misc/LLM_*.md` → `docs/archive/2026-01-azure-performance/`

**Step 1: List all files in misc/**

Use Glob: `docs/misc/*.md`
Expected: 9 Azure/LLM performance files

**Step 2: Create archive destination and move files**

```bash
mkdir -p docs/archive/2026-01-azure-performance
git mv docs/misc/AZURE_*.md docs/archive/2026-01-azure-performance/
git mv docs/misc/RETRYABLE_*.md docs/archive/2026-01-azure-performance/
git mv docs/misc/PROFILING_*.md docs/archive/2026-01-azure-performance/
git mv docs/misc/LLM_*.md docs/archive/2026-01-azure-performance/
```

**Step 3: Check if misc/ is now empty**

Use Glob: `docs/misc/*`
If empty, remove the directory:
```bash
rmdir docs/misc
```

If not empty, leave remaining files for manual review.

**Step 4: Commit**

```bash
git add -A && git commit -m "docs: archive Azure performance analysis docs

Move implementation work-product from misc/ to dated archive.
These document fixes that are now complete (retry logic, memory leaks).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 2: Content Deduplication

### Task 2.1: Create Central Environment Configuration Reference

**Goal:** Extract duplicated environment/secrets documentation into a single reference file, then update other docs to link to it.

**Files:**
- Create: `docs/reference/environment-variables.md`
- Modify: `docs/USER_MANUAL.md` - remove duplicate content, add link
- Modify: `docs/guides/docker.md` - remove duplicate content, add link
- Modify: `docs/reference/configuration.md` - remove duplicate content, add link

**Step 1: Read the authoritative content**

Use Read: `docs/USER_MANUAL.md` (find the Environment Configuration section)
Use Read: `docs/guides/docker.md` (find the Environment Variables section)
Use Read: `docs/reference/configuration.md` (find environment-related content)

Identify the most complete version to use as the base.

**Step 2: Create the consolidated reference**

Use Write to create `docs/reference/environment-variables.md`:

```markdown
# Environment Variables Reference

This is the authoritative reference for ELSPETH environment configuration.

## Required Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `ELSPETH_FINGERPRINT_KEY` | HMAC key for secret fingerprinting | 32+ character random string |

## LLM Provider Variables

| Variable | Purpose | Required When |
|----------|---------|---------------|
| `AZURE_OPENAI_API_KEY` | Azure OpenAI authentication | Using Azure LLM plugins |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | Using Azure LLM plugins |
| `OPENROUTER_API_KEY` | OpenRouter authentication | Using OpenRouter plugins |

## Database Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `ELSPETH_DATABASE_URL` | Database connection string | `sqlite:///./runs/audit.db` |

## .env File Loading

ELSPETH automatically loads `.env` files from the pipeline directory...

[Include the most complete version of this documentation from the sources]
```

**Step 3: Update USER_MANUAL.md**

Use Read to find the exact lines of the Environment Configuration section.
Use Edit to replace the detailed section with a brief intro + link:

```markdown
## Environment Configuration

See [Environment Variables Reference](reference/environment-variables.md) for the complete list.

**Quick start:** Copy `.env.example` to `.env` and fill in your API keys.
```

**Step 4: Update docker.md similarly**

Use Edit to replace duplicated environment section with link to the reference.

**Step 5: Update configuration.md similarly**

Use Edit to add cross-reference to the environment variables doc.

**Step 6: Verify no broken references**

Use Grep: Search for `ELSPETH_FINGERPRINT_KEY` across all docs
Ensure at least one hit in the new reference file.

**Step 7: Commit**

```bash
git add -A && git commit -m "docs: consolidate environment variable documentation

Create single authoritative reference in docs/reference/environment-variables.md.
Update USER_MANUAL, docker guide, and configuration ref to link to it.
Eliminates ~3 duplicated sections.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2.2: Consolidate CLI Command Reference

**Goal:** Ensure CLI commands are documented in one place with other docs linking to it.

**Files:**
- Audit: `docs/USER_MANUAL.md` (CLI Commands section)
- Audit: `docs/guides/docker.md` (Common Commands section)
- Audit: `docs/reference/configuration.md` (any CLI references)

**Step 1: Identify the most complete CLI reference**

Use Read: `docs/USER_MANUAL.md` - find CLI Commands section
Use Read: `docs/guides/docker.md` - find command examples

Compare completeness. USER_MANUAL should be the authoritative source.

**Step 2: Check for undocumented commands**

Use Grep in docs/: Search for `elspeth health`
If found in usage but not in CLI reference, add it.

Use Grep: Search for `--force-new`, `--profile`, `--no-dotenv`
Document any undocumented flags.

**Step 3: Update USER_MANUAL.md with missing commands**

Use Edit to add any missing CLI commands to the reference table.

**Step 4: Update docker.md to reference USER_MANUAL**

Use Edit to add note: "For complete CLI reference, see [User Manual - CLI Commands](../USER_MANUAL.md#cli-commands)"

**Step 5: Commit**

```bash
git add -A && git commit -m "docs: consolidate CLI command documentation

Add missing commands (health) and flags to USER_MANUAL reference.
Add cross-references from docker guide.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2.3: Deduplicate Troubleshooting Sections

**Goal:** Create unified troubleshooting guide, remove duplicated troubleshooting from individual docs.

**Files:**
- Create: `docs/guides/troubleshooting.md`
- Modify: `docs/USER_MANUAL.md` - link to troubleshooting guide
- Modify: `docs/guides/docker.md` - link to troubleshooting guide

**Step 1: Extract troubleshooting content**

Use Read: `docs/USER_MANUAL.md` (Troubleshooting section)
Use Read: `docs/guides/docker.md` (Troubleshooting section)

Compile unique issues from both sources.

**Step 2: Create consolidated troubleshooting guide**

Use Write to create `docs/guides/troubleshooting.md`:

```markdown
# Troubleshooting Guide

## Common Errors

### Secret Fingerprinting Errors

**Error:** `Secret field found but ELSPETH_FINGERPRINT_KEY is not set`

**Cause:** Pipeline uses secrets (API keys) but fingerprinting key not configured.

**Solution:**
1. Generate a key: `openssl rand -hex 32`
2. Add to `.env`: `ELSPETH_FINGERPRINT_KEY=<your-key>`
3. Or set environment variable directly

### Plugin Not Found

**Error:** `Unknown plugin: xyz`

**Cause:** Plugin not installed or misspelled.

**Solution:**
1. Check available plugins: `elspeth plugins list`
2. Verify spelling in configuration
3. Install plugin pack if needed: `uv pip install -e ".[llm]"`

### Docker-Specific Issues

[Docker-specific troubleshooting from docker.md]

### Database Issues

[Database troubleshooting content]
```

**Step 3: Update source docs to link to troubleshooting**

Use Edit on `docs/USER_MANUAL.md`:
Replace detailed troubleshooting with: "See [Troubleshooting Guide](guides/troubleshooting.md)"

Use Edit on `docs/guides/docker.md`:
Keep Docker-specific tips, add link to main troubleshooting guide.

**Step 4: Commit**

```bash
git add -A && git commit -m "docs: create unified troubleshooting guide

Consolidate troubleshooting from USER_MANUAL and docker guide.
Reduces duplication and provides single reference point.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 3: Quality Improvements

### Task 3.1: Create Documentation Navigation Hub

**Goal:** Create `docs/README.md` as the entry point for all documentation.

**Files:**
- Create: `docs/README.md`

**Step 1: Survey existing documentation structure**

Use Glob: `docs/*` (top-level only)
Use Glob: `docs/**/*.md` | count

Build mental map of documentation hierarchy.

**Step 2: Create the navigation hub**

Use Write to create `docs/README.md`:

```markdown
# ELSPETH Documentation

## Quick Start

- **[User Manual](USER_MANUAL.md)** - Installation, configuration, running pipelines
- **[Your First Pipeline](guides/your-first-pipeline.md)** - Step-by-step tutorial
- **[Docker Guide](guides/docker.md)** - Container deployment

## Reference

- **[Configuration Reference](reference/configuration.md)** - All YAML settings
- **[Environment Variables](reference/environment-variables.md)** - API keys, database URLs
- **[Plugin Protocol](contracts/plugin-protocol.md)** - Plugin development

## Operations

- **[Runbooks](runbooks/index.md)** - Operational procedures
- **[Troubleshooting](guides/troubleshooting.md)** - Common errors and solutions
- **[Landscape MCP](guides/landscape-mcp-analysis.md)** - Audit database analysis

## Architecture

- **[System Design](design/architecture.md)** - Core architecture
- **[Requirements Matrix](design/requirements.md)** - Feature status
- **[ADRs](design/adr/)** - Architecture decisions
- **[Subsystems](design/subsystems/)** - Component deep-dives

## Quality & Testing

- **[Test System](TEST_SYSTEM.md)** - Testing strategy
- **[Quality Audit](quality-audit/)** - Code quality findings

## Project Management

- **[Plans](plans/)** - Implementation roadmaps
- **[Bugs](bugs/)** - Issue tracking
- **[Release](release/)** - RC-1 checklists

## Archive

- **[Archived Docs](archive/)** - Historical analyses and completed work
```

**Step 3: Commit**

```bash
git add docs/README.md && git commit -m "docs: create documentation navigation hub

Provides single entry point for all ELSPETH documentation.
Organizes docs by purpose: quick start, reference, operations, architecture.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3.2: Add Resolution Notes to Completed Plans

**Goal:** Add "Implementation Summary" sections to completed plans that lack them.

**Files:**
- Modify: Multiple files in `docs/plans/completed/`

**Step 1: Identify plans lacking resolution notes**

Use Glob: `docs/plans/completed/*.md`
For each file, use Grep to check for "Implementation Summary" or "Resolution" section.

List files that lack this section.

**Step 2: For each plan without resolution, add minimal closure**

Use Read to understand what the plan accomplished.
Use Edit to add section at end:

```markdown
---

## Implementation Summary

**Status:** Completed [DATE]
**Commits:** [List relevant commit hashes if known, or "See git history"]
**Notes:** [Brief description of what was implemented vs. planned]
```

**Step 3: Commit after each 5 plans updated**

```bash
git add docs/plans/completed/ && git commit -m "docs: add resolution notes to completed plans (batch N)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3.3: Create Superseded Plans README

**Goal:** Document what superseded the 12 schema validation attempts in `docs/plans/superseded/`.

**Files:**
- Create: `docs/plans/superseded/schema-validation-attempts-jan24/README.md`

**Step 1: Understand what's in the superseded folder**

Use Glob: `docs/plans/superseded/**/*.md`
Use Read: First few superseded plan files to understand context.

**Step 2: Create README explaining the supersession**

Use Write to create the README:

```markdown
# Superseded: Schema Validation Attempts (January 2024)

## Overview

This folder contains 12 planning documents from early schema validation attempts
that were superseded by a revised approach.

## Why These Were Superseded

[Explanation based on reading the plans - what approach didn't work, what replaced it]

## Current Approach

See: `docs/design/adr/003-schema-validation-lifecycle.md`

## Contents

| File | Original Goal | Superseded Because |
|------|---------------|-------------------|
| [List each file with brief description] |
```

**Step 3: Commit**

```bash
git add docs/plans/superseded/ && git commit -m "docs: add README explaining superseded schema validation plans

Provides context for why 12 plans were superseded and what replaced them.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3.4: Fix Plans README Accuracy

**Goal:** Correct the active plan count and other inaccuracies in `docs/plans/README.md`.

**Files:**
- Modify: `docs/plans/README.md`

**Step 1: Count actual plans by status**

Use Glob: `docs/plans/in-progress/*.md` | count
Use Glob: `docs/plans/paused/*.md` | count
Use Glob: `docs/plans/completed/*.md` | count
Use Glob: `docs/plans/cancelled/*.md` | count

**Step 2: Read and update the README**

Use Read: `docs/plans/README.md`
Use Edit: Update counts to match reality.

**Step 3: Commit**

```bash
git add docs/plans/README.md && git commit -m "docs: fix plans README with accurate counts

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3.5: Rename Audit Folder for Clarity

**Goal:** Rename `docs/audit/` to `docs/audit-trail/` to distinguish from `quality-audit/`.

**Files:**
- Move: `docs/audit/` → `docs/audit-trail/`
- Update: Any references to `docs/audit/`

**Step 1: Check for references to docs/audit/**

Use Grep: Search for `docs/audit/` or `](audit/` across all docs.
Note files that need updating.

**Step 2: Rename the folder**

```bash
git mv docs/audit docs/audit-trail
```

**Step 3: Update references**

Use Edit on each file found in Step 1 to update paths.

**Step 4: Update docs/README.md navigation**

Use Edit to change `audit/` references to `audit-trail/`.

**Step 5: Commit**

```bash
git add -A && git commit -m "docs: rename audit/ to audit-trail/ for clarity

Distinguishes token outcome documentation (audit-trail/) from
code quality findings (quality-audit/).

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Phase 4: Verification

### Task 4.1: Final Structure Verification

**Goal:** Verify the cleanup achieved the intended structure.

**Step 1: Verify no orphaned dated folders**

Use Glob: `docs/*2026*` (should only find files, not folders at top level)
Expected: Empty or only `docs/archive/` contains dated content.

**Step 2: Verify archive consolidation**

Use Glob: `docs/archive/*/`
Expected: Dated folders properly archived.

**Step 3: Verify no duplicate architecture folders**

Use Glob: `docs/architecture/`
Expected: Does not exist (consolidated into design/).

**Step 4: Verify analysis folder removed**

Use Glob: `docs/analysis/`
Expected: Does not exist (moved to quality-audit/).

**Step 5: Verify misc folder cleaned**

Use Glob: `docs/misc/`
Expected: Does not exist or contains only non-Azure content.

**Step 6: Document results**

Create brief verification report noting any issues found.

---

### Task 4.2: Link Integrity Check

**Goal:** Find and fix broken internal documentation links.

**Step 1: Find all internal links**

Use Grep: Search for `](` patterns in `docs/**/*.md`
Extract unique link targets.

**Step 2: Verify each link target exists**

For each unique target path, use Glob to verify the file exists.
List broken links.

**Step 3: Fix broken links**

Use Edit to correct each broken link found.

**Step 4: Commit fixes**

```bash
git add -A && git commit -m "docs: fix broken internal links after restructure

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Summary

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| Phase 1: Structural | 6 tasks | 1-2 hours |
| Phase 2: Deduplication | 3 tasks | 1-2 hours |
| Phase 3: Quality | 5 tasks | 2-3 hours |
| Phase 4: Verification | 2 tasks | 30 min |

**Total: 16 tasks, ~5-7 hours of work**

Each task is designed to be executed by a subagent independently, with clear verification steps and atomic commits.
