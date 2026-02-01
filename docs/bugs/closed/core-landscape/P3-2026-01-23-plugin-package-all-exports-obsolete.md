# Bug Report: Plugin package __all__ exports are obsolete (replaced by PluginManager)

## Summary

All plugin package `__init__.py` files maintain `__all__` exports lists, but these exports are never used. The universal plugin discovery system (PluginManager + dynamic discovery) replaced direct imports completely, making package-level exports dead code.

## Severity

- Severity: minor (cosmetic, maintenance burden)
- Priority: P3

## Reporter

- Name or handle: Claude (via systematic analysis)
- Date: 2026-01-23
- Related run/issue ID: Sprint 1 review

## Environment

- Commit/branch: main
- OS: All
- Python version: 3.12+
- Config profile / env vars: N/A
- Data set or fixture: N/A

## Steps To Reproduce

1. Search entire codebase for package-level imports:
   ```bash
   grep -r "from elspeth.plugins.sources import\|from elspeth.plugins.sinks import\|from elspeth.plugins.transforms import\|from elspeth.plugins.llm import\|from elspeth.plugins.azure import" src/ tests/
   ```
2. Observe: ZERO matches (no usage anywhere)

## Expected Behavior

- Dead code should be removed to reduce maintenance burden
- If exports are needed for public API, they should be documented and tested

## Actual Behavior

Six plugin packages maintain unused `__all__` exports:
- `sources/__init__.py` - exports 3 (complete)
- `sinks/__init__.py` - exports 3 (complete)
- `transforms/__init__.py` - exports 2 (incomplete - missing 5)
- `transforms/azure/__init__.py` - exports 4 (complete)
- `azure/__init__.py` - exports 3 (complete)
- `llm/__init__.py` - exports 12 (complete)

## Evidence

**Zero usage found:**
```bash
# No imports from any plugin package
grep -r "from elspeth.plugins.sources import" src/ tests/  # 0 matches
grep -r "from elspeth.plugins.sinks import" src/ tests/    # 0 matches
grep -r "from elspeth.plugins.transforms import" src/ tests/ # 0 matches
```

**Actual plugin access patterns:**

1. **CLI uses PluginManager:**
   ```python
   manager = _get_plugin_manager()
   manager.register_builtin_plugins()  # Dynamic discovery
   transform = manager.get_transform_by_name("batch_stats")
   ```

2. **Tests import from module paths:**
   ```python
   from elspeth.plugins.transforms.batch_stats import BatchStats
   # NOT: from elspeth.plugins.transforms import BatchStats
   ```

3. **Discovery scans directories automatically:**
   ```python
   # plugins/discovery.py scans *.py files
   # No __all__ needed - finds classes by inheritance
   ```

## Impact

- User-facing impact: None (exports never used)
- Data integrity / security impact: None
- Performance or cost impact: Minimal (extra imports on module load)
- Maintenance burden: Must keep exports in sync as plugins are added

## Root Cause Hypothesis

Legacy code from before PluginManager existed. When dynamic discovery was implemented, direct imports were replaced but old exports were never cleaned up.

## Proposed Fix

**Option A (Cleanest): Delete all exports**
- Remove imports and `__all__` from all 6 plugin `__init__.py` files
- Keep docstrings explaining the package purpose
- If anyone tries to import, Python will give clear error

**Changes:**
1. `sources/__init__.py` - Remove 3 imports + `__all__`
2. `sinks/__init__.py` - Remove 3 imports + `__all__`
3. `transforms/__init__.py` - Remove 2 imports + `__all__`
4. `transforms/azure/__init__.py` - Remove 4 imports + `__all__`
5. `azure/__init__.py` - Remove 3 imports + `__all__`
6. `llm/__init__.py` - Remove 12 imports + `__all__`

**Testing:**
- Run full test suite - should pass (no imports to break)
- Verify CLI still works (uses PluginManager, not imports)

## Architectural Deviations

- Spec or doc reference: N/A (no spec requires these exports)
- Observed divergence: Exports exist but aren't part of any API contract
- Reason (if known): Legacy from pre-PluginManager era
- Alignment plan: Remove dead code

## Acceptance Criteria

- All plugin package `__init__.py` files contain only docstrings
- No `__all__` lists
- No imports (except top-level plugins/__init__.py which is intentionally empty)
- All tests pass

## Tests

- Suggested tests to run: `pytest tests/` (full suite)
- New tests required: No (we're removing unused code, not adding features)

## Notes / Links

- Related: `P3-2026-01-19-transforms-package-exports-incomplete` (superseded by this meta-bug)
- Parent pattern: Phase 6 implementation added PluginManager, obsoleting all direct imports
- Benefits: Cleaner code, less maintenance, clearer API (PluginManager is THE way)

## Subsumes

This meta-bug subsumes/closes:
- P3-2026-01-19-transforms-package-exports-incomplete (incomplete exports don't matter if nothing uses them)
