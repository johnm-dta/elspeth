# Documentation Audit Report — RC-3

**Date:** 2026-02-13
**Branch:** RC3-quality-sprint
**Scope:** Full audit of `docs/` directory (788 files)

---

## Executive Summary

The ELSPETH documentation corpus is **large (788 files), mostly current, and well-organized** — but shows specific weaknesses in release documentation, plugin coverage, and one critical factual error in a contract spec. The bug tracking system is exceptional. The plans lifecycle is well-managed. The architecture docs are comprehensive but have 5 unfulfilled promises for subsystem deep-dives.

**Priority actions:**
- 3 errors of fact — all fixed ✅
- 4 release-blocking gaps (P1)
- 3 duplication clusters to consolidate
- 11 functional gaps to fill

---

## 1. Inventory

| Section | Files | Lines (est.) | Health |
|---------|-------|-------------|--------|
| Top-level (README, USER_MANUAL, TEST_SYSTEM) | 3 | 2,428 | Good |
| design/ (architecture, requirements, ADRs, subsystems) | 11 | 5,000+ | Good (gaps) |
| architecture/ (landscape, telemetry, audit) | 6 | 3,300 | Good (fragmented) |
| guides/ | 7 | 3,243 | Good |
| reference/ | 2 | 1,554 | Good |
| contracts/ | 3 | 2,921 | **1 stale** |
| plugins/ | 1 | 48 | **Severely lacking** |
| runbooks/ | 7 | 1,843 | Good |
| bugs/ (open + closed + archive + process) | ~472 | — | Excellent |
| plans/ (active + completed + paused + superseded) | 173 | — | Good (metadata drift) |
| release/ + release-notes/ | 4 | ~1,200 | **Stale for RC-3** |
| audit-trail/tokens/ | 7 | — | Good |
| analysis/ | 2 | 1,168 | Good |
| testing/ | 3 | 1,035 | Good |
| prompts/ | 3 | 496 | Fine |
| archive/ (3 subdirs) | ~75 | — | Archival |
| performance/ | 1 | 50 | **Minimal** |

---

## 2. Errors of Fact

### ERR-01: `contracts/plugin-protocol.md` stale gate description [FIXED]

**Location:** `docs/contracts/plugin-protocol.md`, line 917 and line 934
**Issue:** The data flow diagram said gates are `(config OR plugin)` — a vestige from before the 2026-02-11 gate plugin removal. Line 934 also claimed gates "may optionally modify row data" which is incorrect (they route tokens, they don't modify data). The rest of the gate section (lines 930-1046) was already correct, describing config-driven gates.
**Impact:** Subtle contradiction — line 917 says "config OR plugin" while line 936 says "NOT plugins."
**Fix applied:** Changed to `(config-driven)`, corrected key property text. Version bumped to v1.10.

### ERR-02: `release/guarantees.md` says "RC-2" where it should say "RC-3" [FIXED]

**Location:** `docs/release/guarantees.md`, lines 242, 248, 292
**Issue:** Three references to "RC-2" despite document header saying "Version: RC-3". Version table only listed RC-2.
**Impact:** Confusion about which release candidate the guarantees apply to.
**Fix applied:** Updated "RC-2" → "RC-3" on lines 242 and 248; added RC-3 row to version table.

### ERR-03: Terminal state lists missing EXPANDED [FIXED]

**Locations:**
- `docs/release/guarantees.md` sections 1.2 and 10 (outcomes table + attributability test)
- `docs/design/adr/002-routing-copy-mode-limitation.md` line 66

**Issue:** EXPANDED (parent token for deaggregation, 1→N expansion) was added in plugin-protocol.md v1.5 (2026-01-19) but never propagated to the guarantees doc or ADR-002. CLAUDE.md correctly lists 9 terminal states including EXPANDED; these docs only listed 7.
**Impact:** A developer checking outcomes against the guarantees doc would not expect EXPANDED tokens.
**Fix applied:** Added EXPANDED to all three locations.

---

## 3. Duplication Clusters

### DUP-01: Archive quality-audit findings ↔ bugs/closed

**Files:** `archive/quality-audit-2026-01-22/` (59 files) vs `bugs/closed/` (456 files)
**Issue:** Many quality-audit findings became formal bug reports. The archive serves as a historical process record, not an active reference.
**Action:** Add a clear "Superseded by docs/bugs/" note to `archive/quality-audit-2026-01-22/README.md`. No content deletion needed (archive is archival by definition).
**Priority:** P3

### DUP-02: Telemetry architecture fragmented across 3 files

**Files:** `architecture/telemetry-emission-points.md` (415 lines), `architecture/telemetry-remediation-plan.md` (603 lines), `architecture/telemetry-implementation-summary.md` (414 lines)
**Issue:** Three documents cover overlapping telemetry concerns (what should be emitted, what's missing, what was implemented). A reader must cross-reference all three.
**Action:** Consolidate into a single `architecture/telemetry.md` with sections for design, emission points, and implementation status.
**Priority:** P2

### DUP-03: Landscape documented at three different levels with no reading order

**Files:** `design/architecture.md` (overview), `design/subsystems/00-overview.md` (subsystem map), `architecture/landscape-system.md` (deep dive)
**Issue:** All three describe Landscape. No reading-order guidance exists.
**Action:** Add a reading-order section to `docs/README.md`.
**Priority:** P3

---

## 4. Release-Blocking Gaps (P1)

### GAP-01: No RC-3 release checklist

**Expected:** `docs/release/rc3-checklist.md`
**Current state:** Only `rc2-checklist.md` exists (historical)
**Source:** Flagged as F-09 in RC3-doc-updates.md (P1 — Release Alignment)
**Action:** Create `rc3-checklist.md` using rc2-checklist.md as template, updated for RC-3 scope

### GAP-02: Feature inventory 2+ weeks stale

**File:** `docs/release/feature-inventory.md`
**Issue:** Header says "January 29, 2026". Missing: graceful shutdown (FEAT-05), DROP-mode sentinel handling, ExecutionGraph API refactoring, multiple RC3-quality-sprint fixes.
**Source:** Flagged as F-10 in RC3-doc-updates.md (P1 — Release Alignment)
**Action:** Update header date, add missing features

### GAP-03: Guarantees version table incomplete

**File:** `docs/release/guarantees.md`
**Issue:** Version table only lists RC-2. RC-3 entry needed.
**Action:** Add RC-3 row with date and summary of changes

### GAP-04: No RC-3 release notes

**File:** `docs/release-notes/` — only contains `rc-2-checkpoint-fix.md` (single bug fix)
**Action:** Create `rc-3-release-notes.md` summarizing 65+ completed remediation items

---

## 5. Functional Gaps

### GAP-05: Plugin documentation (1 of 20+ plugins documented) [P2]

**Current:** Only `plugins/web-scrape-transform.md` (48 lines)
**Missing:** CSV/JSON sources and sinks, field_mapper, passthrough, truncate, json_explode, LLM transforms (Azure OpenAI, OpenRouter, batch, multi-query), content safety, prompt shield, database sink, blob sink/source, batch_stats, batch_replicate, keyword_filter
**Action:** Create a plugin catalog with config schemas, input/output contracts, and examples

### GAP-06: 5 missing subsystem deep-dives [P2]

**Current:** Only Landscape (landscape-system.md) and Token Lifecycle (06-token-lifecycle.md) exist
**Missing:** Plugin System, SDA Engine, Configuration, Payload Store, CLI
**Source:** Promised in `design/subsystems/00-overview.md`
**Action:** Write subsystem docs or explicitly de-scope in overview

### GAP-07: No LLM pipeline guide [P2]

**Issue:** The primary use case (LLM-powered classification/analysis pipelines) has no dedicated how-to guide
**Action:** Create `guides/llm-pipelines.md` covering Azure OpenAI, multi-query, batch processing, content safety, rate limiting

### GAP-08: Performance documentation minimal [P3]

**Current:** Single `performance/schema-refactor-baseline.md` (50 lines, 1 scenario)
**Missing:** Row processing throughput, memory profiles, checkpoint timing, large pipeline benchmarks
**Action:** Expand with baselines from Phase 6 performance tests

### GAP-09: No performance/scaling runbook [P3]

**Missing:** Operational guidance for tuning concurrency, rate limits, database optimization under load
**Action:** Create `runbooks/performance-tuning.md`

### GAP-10: `docs/code_analysis/` referenced in MEMORY.md but doesn't exist [P3]

**Issue:** MEMORY.md references deliverables (`_repair_manifest.md`, `_verdicts.csv`) stored in `docs/code_analysis/` — directory not found
**Action:** Locate via git history or update MEMORY.md reference

### GAP-11: No reading-order guide for newcomers [P3]

**Issue:** 788 files with no "start here" navigation beyond docs/README.md link list
**Action:** Add reading-order section to README.md (e.g., "CLAUDE.md → architecture.md → subsystems overview → ADRs → deep dives")

---

## 6. Organizational Issues

### ORG-01: RC3-doc-updates.md not moved to completed/ [P2]

**File:** `docs/plans/RC3-doc-updates.md`
**Issue:** Marked "IMPLEMENTED (2026-02-13)" on line 4 but still at root level
**Action:** Move to `docs/plans/completed/` or verify implementation is actually complete

### ORG-02: rc2-checklist.md still in active release/ directory [P3]

**File:** `docs/release/rc2-checklist.md`
**Issue:** Historical RC-2 document in active directory
**Action:** Keep as reference (useful as RC-3 template) but create rc3-checklist.md alongside it

### ORG-03: plans/README.md undercounts superseded plans [P3]

**File:** `docs/plans/README.md`
**Issue:** Claims "4 superseded plans" but actual count is 17 (3 main + 14 schema-validation-attempts)
**Action:** Update count or clarify that schema-validation subfolder is a separate archive

### ORG-04: RC3-remediation.md internal contradiction [P2]

**File:** `docs/plans/RC3-remediation.md`
**Issue:** FEAT-05 (Graceful Shutdown) marked DONE on line 78 but still included in effort estimates (lines 163, 174). Effort totals are overstated by 5-10 days.
**Action:** Remove FEAT-05 from remaining work totals and priority ordering

### ORG-05: Feature inventory header date stale [P1]

**File:** `docs/release/feature-inventory.md`
**Issue:** Header says "January 29, 2026" despite file being touched recently
**Action:** Update to current date after content refresh (see GAP-02)

### ORG-06: TEST_SYSTEM.md has stale paths [P3]

**File:** `docs/TEST_SYSTEM.md`
**Issue:** Some references still use `tests_v2/` paths despite Phase 7 cutover renaming to `tests/`
**Action:** Search-and-replace `tests_v2/` → `tests/`

---

## 7. Enhancement Opportunities

### ENH-01: Consolidate telemetry architecture docs [P2]

Merge `telemetry-emission-points.md`, `telemetry-remediation-plan.md`, and `telemetry-implementation-summary.md` into a single `architecture/telemetry.md` with clear sections.

### ENH-02: Add "Next Steps" links between guides [P3]

`your-first-pipeline.md` → LLM pipeline guide → troubleshooting → runbooks

### ENH-03: Create plugin catalog [P2]

Single-page reference listing all built-in plugins with config schemas, I/O contracts, and example YAML snippets.

### ENH-04: Add reading-order guide to docs/README.md [P3]

Tell newcomers: CLAUDE.md → architecture.md → subsystems/00-overview.md → ADRs → deep dives → guides

### ENH-05: Archive quality-audit with supersession note [P3]

Add README note to `archive/quality-audit-2026-01-22/` explaining it's superseded by `bugs/`.

---

## 8. Work Priority Summary

| Priority | Count | Items |
|----------|-------|-------|
| **Fix now** | 3 | ERR-01 (plugin-protocol gate desc) ✅, ERR-02 (guarantees RC-2 refs) ✅, ERR-03 (missing EXPANDED state) ✅ |
| **P1** (release-blocking) | 4 | GAP-01 (RC-3 checklist), GAP-02 (feature inventory), GAP-03 (version table), GAP-04 (release notes) |
| **P2** (consistency) | 6 | DUP-02, GAP-05, GAP-06, GAP-07, ORG-01, ORG-04 |
| **P3** (polish) | 10 | DUP-01, DUP-03, GAP-08-11, ORG-02-03, ORG-05-06 |
| **Enhancement** | 5 | ENH-01 through ENH-05 |
