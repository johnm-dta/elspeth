# Test Suite Quality Audit Plan

## Objective

Conduct a comprehensive, file-by-file audit of the ELSPETH test suite (200 files) to identify:

1. **Poorly Constructed Tests**
   - Sleepy assertions (tests that pass even when they should fail)
   - Defective or deficient tests
   - Obvious incompleteness (mutation vulnerabilities)
   - Missing edge cases or error handling

2. **Misclassified Tests**
   - Unit tests masquerading as integration tests
   - Integration tests that should be unit tests
   - Tests in wrong files or organizational structure

3. **Infrastructure Gaps**
   - Tests that need better scaffolding
   - Repeated setup code that should be fixtures
   - Missing test data management
   - Poor isolation or test interdependence

## Process

### Phase 1: Discovery & Setup ✅
- Enumerate all 200 test files
- Create workspace: `docs/quality-audit/findings/<subsystem>/`
- Group files by subsystem

### Phase 2: Batched Agent Review (In Progress)
- Launch `ordis-quality-engineering:test-suite-reviewer` agents in batches of 5-6 files
- Each agent:
  1. Reads `CLAUDE.md` for project context
  2. Reviews ONE test file against best practices
  3. Writes findings to `findings/<subsystem>/<filename>.md`
  4. No compromises - flag everything, even if tedious to fix

### Phase 3: Collation
- Read all findings files
- Identify cross-cutting patterns
- Categorize by severity and effort

### Phase 4: Action Plan
- Generate prioritized backlog:
  - **P0 Critical**: Tests that don't test anything
  - **P1 High**: Misclassified tests, major gaps
  - **P2 Medium**: Infrastructure improvements
  - **P3 Low**: Style and consistency

## Success Criteria

- Every test file reviewed against best practices
- Findings documented with file:line references
- Actionable recommendations with clear priority
- No "good enough" compromises

## Timeline

- Started: 2026-01-25
- Estimated batches: ~35 (200 files ÷ 6 per batch)
- Target completion: Today

## Context for Agents

Agents must read and internalize CLAUDE.md, especially:
- **Auditability Standard**: High-stakes accountability, no inference
- **Three-Tier Trust Model**: Full trust (our data), elevated trust (pipeline), zero trust (external)
- **No Bug-Hiding Patterns**: Direct field access, crash on anomalies
- **Terminal Row States**: Every row reaches exactly one terminal state
