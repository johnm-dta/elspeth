# Schema Validation Implementation Attempts - January 24-25, 2026

This directory contains **13 implementation plans** created during a 2-day period (Jan 24-25, 2026) while solving a critical schema validation architecture issue.

## What Happened

During RC-1 development, schema validation was discovered to be completely non-functional. Multiple implementation approaches were attempted, each partially implemented before hitting integration bugs that forced a replan.

## The Problem

**Root cause:** Schema validation needed to happen pre-instantiation (during graph construction), but plugins set schemas in `__init__()` post-instantiation. This temporal mismatch created several attempts to solve it:

1. Extract schemas from config models (but configs don't have schemas)
2. Introspect Pydantic models for dynamic schemas (but caused detection regressions)
3. Instantiate plugins earlier (but broke separation of concerns)
4. Add schema inheritance to gates (but still left gaps)
5. Multiple architectural refactor attempts (each hitting different showstoppers)

## Timeline of Attempts

| Date | Plan | Approach | Outcome |
|------|------|----------|---------|
| Jan 24 | eliminate-parallel-schema-detection | Remove Pydantic introspection | ❌ Marked OBSOLETE in file |
| Jan 24 | fix-schema-validation-bypass | Use PluginManager for class schemas | ❓ Partial implementation |
| Jan 24 | fix-schema-validation-architecture | Move validation earlier | ❓ Partial implementation |
| Jan 24 | fix-schema-validation-properly | Comprehensive fix with reviews | ❓ Partial, integration bug hit |
| Jan 24 | schema-inheritance-for-gates | Gates inherit upstream schemas | ❓ Partial implementation |
| Jan 24 | schema-refactor-00-overview | 4-phase refactor plan | ❓ Overview only |
| Jan 24 | schema-refactor-01-foundation | Phase 1: Foundation changes | ❓ Partial implementation |
| Jan 24 | schema-refactor-02-cli-refactor | Phase 2: CLI changes | ❓ Partial implementation |
| Jan 24 | schema-refactor-03-testing | Phase 3: Test updates | ❓ Partial implementation |
| Jan 24 | schema-refactor-04-cleanup | Phase 4: Cleanup | ❓ Not started |
| Jan 24 | schema-validation-architectural-refactor | Architectural approach v1 | ❓ Partial implementation |
| Jan 24 | schema-validation-architectural-refactor-v2 | Architectural approach v2 | ❓ Partial implementation |
| Jan 25 | redesign-validation-enforcement | Fix enforcement mechanism | ❓ Partial, may be superseded |

## Final Solution

**Architectural Decision:** [ADR 003: Schema Validation Lifecycle](../../../design/adr/003-schema-validation-lifecycle.md)

**Implementation Plan:** [Validation Subsystem Extraction](../completed/2026-01-25-validation-subsystem-extraction.md)

**Approach:** Restructure CLI to instantiate plugins BEFORE graph construction, then build the graph from plugin instances using `ExecutionGraph.from_plugin_instances()`. This eliminates the temporal mismatch by making schemas available when the graph is constructed.

**Status:** Implemented and accepted

## Git History Evidence

Key commits showing the iterative process:

```bash
25d5a6e feat: add schema configuration validation
3704012 feat: add validation for transforms, gates, and sinks
1b38d50 feat: add plugin config validation subsystem
efa8ca3 docs: define phased rollout strategy for validation migration
8809bd1 feat: add edge compatibility validation to ExecutionGraph
df43269 refactor: remove schema validation from DAG layer
430307d feat: add schema validation to plugin protocols
29e233f docs: mark schema detection bugs as resolved
```

## Why These Plans Are Preserved

These plans document:
- **Architectural decision-making process** - showing what approaches were tried and why they didn't work
- **Integration challenges** - each plan hit specific bugs that weren't visible until implementation
- **Evolution of understanding** - the problem's true nature became clearer with each attempt
- **Review feedback** - 4-expert reviews revealed critical gaps in earlier approaches

## Related Bugs

The schema validation crisis created several P0/P1 bugs (now at `docs/bugs/` root):
- `P0-2026-01-24-schema-validation-non-functional.md`
- `P0-2026-01-24-dynamic-schema-detection-regression.md`
- `P0-2026-01-24-eliminate-parallel-dynamic-schema-detection.md`
- `P2-2026-01-24-aggregation-nodes-lack-schema-validation.md`
- `P3-2026-01-24-coalesce-nodes-lack-schema-validation.md`

Several related bugs were closed during this process (in `docs/bugs/closed/`):
- `P1-2026-01-21-schema-validator-ignores-dag-routing.md`
- `P1-2026-01-19-shape-changing-transforms-output-schema-mismatch.md`
- `P1-2026-01-20-schema-compatibility-check-fails-on-optional-and-any.md`

## Lessons Learned

1. **Pre-instantiation validation is hard** when schemas are set in constructors
2. **Architectural mismatches** (temporal, ownership) require architectural solutions, not workarounds
3. **Integration testing is critical** - several "complete" plans failed only during integration
4. **Iterative problem-solving** sometimes requires multiple attempts before finding the right solution
5. **Code review matters** - 4-expert reviews caught critical gaps in proposed solutions

## For Future Reference

If modifying validation or schema handling:
1. Read the ADR first: [ADR 003: Schema Validation Lifecycle](../../../design/adr/003-schema-validation-lifecycle.md)
2. Review the implementation plan: [Validation Subsystem Extraction](../completed/2026-01-25-validation-subsystem-extraction.md)
3. Review these attempts to understand **what doesn't work** and why
4. Check git history for the actual implementation commits
5. Verify related bugs are still closed after changes

---

**Archived:** 2026-01-25
**Context:** Iterative problem-solving during schema validation crisis (Jan 24-25, 2026)
**Final Solution:** [ADR 003: Schema Validation Lifecycle](../../../design/adr/003-schema-validation-lifecycle.md)
