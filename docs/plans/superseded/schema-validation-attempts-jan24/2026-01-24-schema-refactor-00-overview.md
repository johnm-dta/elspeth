# Schema Validation Architectural Refactor - Overview

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

## Plan Structure

This implementation plan is split into multiple files to work around context limitations:

1. **00-overview.md** (this file) - Summary and design
2. **01-foundation.md** - Tasks 1-4: PluginManager fix, helper functions, graph construction
3. **02-cli-refactor.md** - Tasks 5-7: CLI commands (run, validate, resume)
4. **03-testing.md** - Tasks 8-10: Comprehensive tests and validation
5. **04-cleanup.md** - Tasks 11-15: Documentation, bug closure, final cleanup

## Quick Reference

**Goal:** Fix schema validation by instantiating plugins before graph construction

**Timeline:** 4-5 days (15 tasks)

**Blockers Fixed:**
- P0-2026-01-24-schema-validation-non-functional
- P2-2026-01-24-aggregation-nodes-lack-schema-validation
- P3-2026-01-24-coalesce-nodes-lack-schema-validation

## Design Overview

### Current Flow (Broken)
```
1. Load config (Pydantic models)
2. Build graph from config (schemas unavailable)
3. Validate graph (all schemas None, validation skipped)
4. Instantiate plugins (schemas NOW available, but too late)
5. Execute pipeline
```

### New Flow (Fixed)
```
1. Load config (Pydantic models)
2. Instantiate plugins (schemas available on instances)
3. Build graph from plugin instances (extract schemas directly)
4. Validate graph (schemas populated, validation works)
5. Execute pipeline (use already-instantiated plugins)
```

## Key Architectural Changes

| Component | Change | Impact |
|-----------|--------|--------|
| **PluginManager** | Raise exceptions on missing plugins | No defensive None checks (CLAUDE.md compliant) |
| **cli_helpers.py** | New module with `instantiate_plugins_from_config()` | Plugins created before graph |
| **ExecutionGraph** | Add `from_plugin_instances()` classmethod | Schema extraction from instances |
| **ExecutionGraph** | Remove `from_config()` entirely | Clean API, no legacy code |
| **Validation** | Handle aggregation dual-schemas | Incoming uses input_schema, outgoing uses output_schema |
| **CLI commands** | Refactor run/validate/resume | Use new construction flow |
| **Execution** | `_execute_pipeline_with_instances()` | Reuse plugins (no double instantiation) |

## Critical Issues Fixed from Multi-Agent Review

### 1. Coalesce Implementation Missing ✅
**Review Finding:** "Coalesce implementation completely missing - any pipeline using fork/coalesce patterns will fail"

**Fix:** Task 3 includes complete coalesce implementation (lines 180-230) with:
- `_coalesce_id_map` and `_branch_to_coalesce` population
- Node creation with proper config
- Edge creation from fork gates to coalesce nodes
- Edge creation from coalesce to output sink

### 2. Task 5 Implementation Truncated ✅
**Review Finding:** "Plan says 'copy from existing' without providing actual code"

**Fix:** Task 5 (in 02-cli-refactor.md) provides complete implementation with:
- Full event formatter registration code
- Explicit plugin reuse (no double instantiation)
- PipelineConfig construction from instances

### 3. Resume Command Not Addressed ✅
**Review Finding:** "Resume command still uses from_config() - will break after deletion"

**Fix:** Task 7 (in 02-cli-refactor.md) explicitly updates resume command:
- Instantiate plugins before graph reconstruction
- Update `_build_resume_graph_from_db()` to use instances
- Add integration test for checkpoint → resume cycle

### 4. Defensive Programming Violates CLAUDE.md ✅
**Review Finding:** "Plugin lookup returning None is defensive pattern that hides bugs"

**Fix:** Task 1 makes PluginManager raise exceptions:
- No `if cls is None:` checks
- ValueError with available plugins listed
- Aligns with "No Bug-Hiding Patterns" policy

### 5. Test Coverage Gaps ✅
**Review Finding:** "40% coverage - missing error handling, edge direction, regression tests"

**Fix:** Task 8 (in 03-testing.md) adds:
- Plugin instantiation error tests
- Aggregation edge direction tests (incoming vs outgoing)
- Runtime failure prevention tests
- Regression test proving old bug is fixed

### 6. Deprecation Violates No-Legacy Policy ✅
**Review Finding:** "Plan deprecates from_config() then removes later - creates technical debt"

**Fix:** Task 11 (in 04-cleanup.md) deletes `from_config()` immediately:
- No deprecation period
- All callers updated in Tasks 5-7
- Clean removal per CLAUDE.md

## Task Summary

### Foundation (Tasks 1-4) - File: 01-foundation.md
- **Task 1:** Fix PluginManager to raise on missing plugins
- **Task 2:** Add `instantiate_plugins_from_config()` helper
- **Task 3:** Add `ExecutionGraph.from_plugin_instances()` with COMPLETE coalesce
- **Task 4:** Update `_validate_edge_schemas()` for aggregation dual-schema

### CLI Refactor (Tasks 5-7) - File: 02-cli-refactor.md
- **Task 5:** Refactor `run()` command + add `_execute_pipeline_with_instances()`
- **Task 6:** Refactor `validate()` command
- **Task 7:** Update `resume()` command (CRITICAL - was missing in v1)

### Testing (Tasks 8-10) - File: 03-testing.md
- **Task 8:** Comprehensive integration tests + error handling tests
- **Task 9:** Regression prevention test (prove old bug is fixed)
- **Task 10:** Run full test suite and fix regressions

### Cleanup (Tasks 11-15) - File: 04-cleanup.md
- **Task 11:** Delete `from_config()` entirely (no deprecation)
- **Task 12:** Update documentation and ADR
- **Task 13:** Close bugs (P0, P2, P3)
- **Task 14:** Audit other `getattr()` patterns (follow-up work)
- **Task 15:** Final integration test and verification

## Acceptance Criteria

- [x] Schema validation detects incompatible transform chains
- [x] Schema validation detects incompatible source → transform edges
- [x] Schema validation detects incompatible transform → sink edges
- [x] Schema validation detects aggregation dual-schema edges
- [x] Schema validation handles coalesce fork/join patterns
- [x] Schema validation handles dynamic schemas (`None`) correctly
- [x] Integration tests verify end-to-end validation works
- [x] No double plugin instantiation
- [x] CLI commands (run, validate, resume) use new construction
- [x] Plugin errors caught during validation (fail-fast)
- [x] Resume command works after checkpoint
- [x] Documentation updated (ADR, bug status)
- [x] No regressions in existing tests
- [x] All bugs closed (P0, P2, P3)
- [x] No legacy code (`from_config()` deleted)

## Timeline Estimate

**Per-Task Breakdown:**
- Foundation (Tasks 1-4): 6-8 hours
- CLI Refactor (Tasks 5-7): 8-10 hours (Task 7 resume is complex)
- Testing (Tasks 8-10): 4-6 hours
- Cleanup (Tasks 11-15): 2-4 hours

**Total: 20-28 hours → 3-4 work days**

With code review, debugging, and integration testing: **4-5 days**

## Implementation Approach

**Recommended:** Use `superpowers:subagent-driven-development` for task-by-task execution with review checkpoints.

**Alternative:** Use `superpowers:executing-plans` in a separate session for batch execution.

**Start with:** `docs/plans/2026-01-24-schema-refactor-01-foundation.md`

---

**Next:** Read `01-foundation.md` to begin implementation
