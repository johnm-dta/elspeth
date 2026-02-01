# ELSPETH Bug Tracking

This directory tracks all bugs found in the ELSPETH codebase, organized by subsystem for systematic fixing.

## Directory Structure

```
docs/bugs/
├── open/                  # Active bugs organized by subsystem
│   ├── cli/              # Command-line interface bugs
│   ├── core-config/      # Configuration system bugs
│   ├── core-dag/         # DAG validation and graph construction
│   ├── core-landscape/   # Audit trail, recovery, repositories
│   ├── cross-cutting/    # Multi-subsystem issues (schema, validation)
│   ├── engine-coalesce/  # Fork/join/merge execution
│   ├── engine-orchestrator/ # Pipeline orchestration
│   ├── engine-pooling/   # LLM pooling and batching
│   ├── engine-processor/ # Token management
│   ├── engine-retry/     # Retry logic and backoff
│   ├── engine-spans/     # Observability and tracing
│   ├── llm-azure/        # Azure OpenAI integration
│   ├── plugins-llm/      # Base LLM transforms
│   ├── plugins-sinks/    # Sink implementations
│   ├── plugins-sources/  # Source implementations
│   └── plugins-transforms/ # Transform implementations
│
├── closed/               # Resolved bugs (historical record)
├── plans/                # Bug fix implementation plans
│   └── completed/       # Completed fix plans
│
├── archive/              # Old automation artifacts and obsolete files
│   ├── generated-2026-01-22/     # Automated triage artifacts
│   ├── scripts-obsolete-2026-01-24/ # Old organization scripts
│   └── process-2026-01-22/       # Old process logs
│
├── BUGS.md              # Quick reference bug index
├── BUG-TRIAGE-REPORT-*.md  # Triage session reports
└── VERIFICATION-REPORT-*.md # Bug verification reports
```

## Bug Organization

### By Subsystem (Primary)

Bugs are organized into subsystem-specific directories under `open/`. This enables:
- **Systematic fixing**: Address related bugs together
- **Expert assignment**: Route bugs to subsystem owners
- **Quality insights**: Identify hotspots requiring attention

### Bug Counts by Subsystem

**Triage note (2026-02-01):** Current open-bug counts are maintained in `docs/bugs/open/README.md`. The table below reflects the historical 2026-01-25 verification snapshot.

| Subsystem | P1 | P2 | P3 | Total | Notes |
|-----------|----|----|----|----|-------|
| **llm-azure** | 3 | 8 | 1 | 12 | Azure integration, error handling ⚠️ |
| **engine-orchestrator** | 2 | 4 | 3 | 9 | Aggregation, quarantine, resume |
| **engine-coalesce** | 4 | 4 | 0 | 8 | Fork/join semantics, timeouts ⚠️ |
| **core-landscape** | 3 | 2 | 2 | 7 | Recovery, audit integrity |
| **plugins-sinks** | 3 | 2 | 1 | 6 | Schema validation, mode handling |
| **core-config** | 0 | 2 | 3 | 5 | Config contracts, metadata |
| **engine-pooling** | 0 | 3 | 2 | 5 | LLM pooling and batching |
| **engine-retry** | 0 | 2 | 2 | 4 | Retry logic, backoff |
| **cross-cutting** | 1 | 2 | 1 | 4 | Schema architecture |
| **engine-processor** | 1 | 3 | 0 | 4 | Token management |
| **plugins-transforms** | 2 | 2 | 0 | 4 | Type coercion, batch ops |
| Other subsystems | 3 | 6 | 3 | 12 | Various |
| **TOTAL** | **19** | **36** | **18** | **73** | |

**Hotspots requiring immediate attention:**
- ⚠️ **llm-azure**: 12 bugs, primarily error handling gaps
- ⚠️ **engine-coalesce**: 8 bugs, fork/join semantics broken
- ⚠️ **engine-orchestrator**: 9 bugs, aggregation issues

## Bug Naming Convention

All bugs follow this naming pattern:

```
{PRIORITY}-{DATE}-{SHORT-DESCRIPTION}.md
```

Examples:
- `P0-2026-01-22-source-row-payloads-never-persisted.md`
- `P1-2026-01-21-duplicate-branch-names-break-coalesce.md`
- `P2-2026-01-21-aggregation-input-hash-mismatch.md`

## Priority Levels

| Priority | Severity | Fix Timeline | Count |
|----------|----------|--------------|-------|
| **P0** | Critical - System broken | This week | 1 |
| **P1** | Major - Audit/data integrity | This sprint | 19 |
| **P2** | Moderate - Functionality gaps | Next sprint | 36 |
| **P3** | Minor - Quality/UX issues | Backlog | 18 |

## Bug Lifecycle

```
[Discovered] → open/{subsystem}/
     ↓
[Fixed & Verified]
     ↓
closed/
```

### Bug States

1. **Open**: Active bug in `open/{subsystem}/`
   - Verified against current code
   - Includes reproduction steps
   - Has fix recommendations

2. **Closed**: Resolved bug in `closed/`
   - Includes resolution type (fixed/obe/wontfix)
   - Links to fix commits/PRs
   - Kept for historical reference

## Verification Status

**Triage note (2026-02-01):** The verification stats below are historical. See `docs/bugs/open/README.md` for the latest open-bug totals after duplicate cleanup.

All bugs have been systematically verified against the current codebase:

- ✅ **94% STILL VALID** (66/70 bugs verified) - Real technical debt
- 🔄 **4 OBE** (Overtaken By Events) - Fixed by refactors
- ❌ **0 LOST** - No bugs invalidated by code changes

**Last verification:** 2026-01-25 (see `VERIFICATION-REPORT-2026-01-25.md`)

## Finding Bugs

### By Subsystem
Navigate to `open/{subsystem}/` to see all bugs for that component.

### By Priority
Use grep to filter by priority:
```bash
# All P0/P1 bugs
find docs/bugs/open -name "P[01]-*.md"

# P1 bugs in specific subsystem
ls docs/bugs/open/engine-coalesce/P1-*.md
```

### By Keyword
```bash
# Search bug descriptions
grep -r "audit trail" docs/bugs/open/

# Search by component mentioned in content
grep -r "DatabaseSink" docs/bugs/open/
```

## Creating New Bugs

When filing a new bug:

1. **Verify it's real**: Check current code, not assumptions
2. **Choose priority**: Use the priority table above
3. **Identify subsystem**: Place in correct `open/{subsystem}/` directory
4. **Use naming convention**: `{PRIORITY}-{DATE}-{SHORT-DESCRIPTION}.md`
5. **Include required sections**:
   - Summary
   - Current Behavior
   - Expected Behavior
   - Impact
   - Fix Recommendation (if known)
   - Verification (code references, line numbers)

## Archive

The `archive/` directory contains:
- **generated-2026-01-22/**: Automated triage artifacts from Jan 22
  - CATALOG.md, DUPLICATES.md, TRIAGE_SUMMARY.md
  - Source code analysis mirrors (37 .md files)
  - Old verification reports
- **scripts-obsolete-2026-01-24/**: Organization scripts (no longer needed)
- **process-2026-01-22/**: Old process logs (CODEX_LOG.md)

These are kept for historical reference but are no longer actively used.

## Reports

- **BUGS.md**: Quick reference bug index
- **BUG-TRIAGE-REPORT-*.md**: Reports from systematic triage sessions
- **VERIFICATION-REPORT-*.md**: Bug verification audit results
- **open/README.md**: Detailed subsystem breakdown with fix priorities

## Next Steps

1. **Fix P0 bugs immediately** (1 critical bug blocking audit compliance)
2. **Address P1 bugs this sprint** (19 major bugs affecting data integrity)
3. **Tackle hotspots**:
   - llm-azure (12 bugs)
   - engine-coalesce (8 bugs)
   - engine-orchestrator (9 bugs)

See `open/README.md` for detailed fix priorities and subsystem ownership.
