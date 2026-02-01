# Bug Report: Dynamic schema detection broken after from_plugin_instances() refactor

## Summary

- Dynamic schema detection in `ExecutionGraph.validate()` is broken after the `from_plugin_instances()` refactor
- Root cause: Validation logic checks `if schema is None` to detect dynamic schemas, but schemas are now Pydantic model instances (not `None`)
- Dynamic schemas are Pydantic models with `model_fields == {}` and `model_config['extra'] == 'allow'`
- Result: Dynamic schema validation is NOT being skipped as designed, causing false validation failures
- This bug was introduced as a regression when fixing P0-2026-01-24-schema-validation-non-functional

## Severity

- Severity: **blocker** (dynamic schemas incorrectly validated)
- Priority: **P0**
- Impact: Breaks RC-1 testing - pipelines with dynamic schemas fail validation incorrectly

## Reporter

- Name or handle: systematic-debugging-session (Phase 1-3 investigation complete)
- Date: 2026-01-24
- Related issue ID: Discovered during Plan 3 (Testing Tasks) execution after P0-2026-01-24 fix
- Session: fix/rc1-bug-burndown-session-4

## Environment

- Commit/branch: `fix/rc1-bug-burndown-session-4` @ `778d903`
- OS: Linux 6.8.0-90-generic
- Python version: 3.13.1
- Config profile / env vars: All environments affected
- Data set or fixture: Any pipeline using dynamic schemas (CSV source, passthrough transform, aggregations)

## Steps To Reproduce

1. Create pipeline with dynamic source feeding specific sink:
   ```yaml
   datasource:
     plugin: csv
     options:
       path: input.csv
       schema: {fields: dynamic}  # Dynamic schema

   sinks:
     output:
       plugin: csv
       options:
         path: output.csv
         schema:
           mode: strict
           fields:
             - "value: float"  # Specific schema

   output_sink: output
   ```

2. Run validation:
   ```bash
   elspeth validate --settings pipeline.yaml
   ```

3. **Expected:** Validation PASSES (dynamic schemas skip compatibility checks)
4. **Actual:** Validation would FAIL if not for the fix (dynamic schema not detected as dynamic)

## Expected Behavior

- Dynamic schemas should be detected by introspecting Pydantic model structure
- Validation should skip when producer OR consumer is dynamic
- `test_dynamic_schema_to_specific_schema_validation()` integration test should PASS

## Actual Behavior (Before Fix)

- `if schema is None` check never triggers (schemas are Pydantic instances)
- Dynamic schemas processed through normal validation logic
- `_get_missing_required_fields()` incorrectly reports missing fields for dynamic schemas
- Integration test `test_dynamic_schema_to_specific_schema_validation()` was SKIPPED with bug documentation

## Evidence

### The Refactor That Broke Dynamic Detection

**Before `from_plugin_instances()` refactor (commit 653dd8a):**
```python
# src/elspeth/core/dag.py:455-456 (OLD)
graph.add_node(
    tid,
    input_schema=getattr(plugin_config, "input_schema", None),  # Returns None
    output_schema=getattr(plugin_config, "output_schema", None),  # Returns None
)
```
- Dynamic schemas represented as `None` at class level
- `if schema is None` correctly detected dynamic schemas

**After `from_plugin_instances()` refactor:**
```python
# src/elspeth/core/dag.py:523-531 (NEW - commit 2027906)
transform_cls = manager.get_transform_by_name(plugin_config.plugin)

# Get schemas from class attributes (may be None for dynamic schemas)
input_schema = getattr(transform_cls, "input_schema", None)
output_schema = getattr(transform_cls, "output_schema", None)

graph.add_node(
    tid,
    input_schema=input_schema,   # Now a Pydantic model instance
    output_schema=output_schema,  # Now a Pydantic model instance
)
```
- Schemas extracted from plugin class attributes
- `schema_factory.create_schema_from_config()` creates Pydantic model instances for dynamic schemas
- Dynamic schemas are NO LONGER `None` - they are Pydantic models

### Dynamic Schema Instantiation (src/elspeth/plugins/schema_factory.py:78-91)

```python
def _create_dynamic_schema(name: str) -> type[PluginSchema]:
    """Create a schema that accepts any fields."""
    return create_model(
        name,
        __base__=PluginSchema,
        __module__=__name__,
        __config__=ConfigDict(
            extra="allow",  # Key characteristic
            # No strict setting needed - no fields to validate types against
        ),
    )
```

**Runtime characteristics verified:**
```python
# Dynamic schema:
schema.model_fields == {}           # Empty dict
schema.model_config['extra'] == 'allow'  # Accepts any fields

# Explicit schema:
schema.model_fields == {'value': FieldInfo(...)}  # Has fields
schema.model_config['extra'] == 'forbid'          # Rejects extra
```

### Broken Validation Logic (src/elspeth/core/dag.py:293-295)

```python
# Skip validation if either schema is None (dynamic)
if producer_schema is None or consumer_schema is None:
    continue  # ❌ NEVER TRIGGERS - schemas are Pydantic instances now
```

**Impact on downstream validation:**
```python
# src/elspeth/core/dag.py:298-301
missing = _get_missing_required_fields(
    producer=producer_schema,  # Dynamic schema with empty model_fields
    consumer=consumer_schema,   # Specific schema requiring fields
)

# _get_missing_required_fields() logic:
producer_fields = set(producer.model_fields.keys())  # Empty set for dynamic!
required_fields = {name for name, field in consumer.model_fields.items() if field.is_required()}
return required_fields - producer_fields  # Returns ALL required fields (incorrect!)
```

**Result:** Dynamic → Specific edges incorrectly report "producer missing required fields"

### Skipped Integration Test (tests/cli/test_plugin_errors.py:364-379)

```python
@pytest.mark.skip(
    reason="Dynamic schema detection broken after from_plugin_instances() refactor. "
    "Schemas are now instances (not None), need to detect dynamic via empty model_fields. "
    "Bug tracked separately."
)
def test_dynamic_schema_to_specific_schema_validation():
    """Test validation behavior when dynamic schema feeds specific schema.

    CRITICAL GAP from review: Undefined behavior for dynamic → specific.

    Expected behavior (documented): Dynamic schemas skip validation (pass-through).

    BUG: After refactoring to from_plugin_instances(), schemas are instantiated
    so they're never None. Need to update validation logic to detect dynamic schemas
    by checking model_fields == {} and model_config['extra'] == 'allow'.
    """
    # Test case: Dynamic source → Specific sink (should PASS - validation skipped)
    # Currently FAILS because dynamic not detected
```

## Impact

### User-facing impact
- Pipelines using CSV source (dynamic schema) fail validation incorrectly
- Passthrough transforms with dynamic schemas rejected
- Aggregations with intentional dynamic output schemas fail validation
- **Blocks RC-1 testing:** Cannot test real-world pipelines with dynamic schemas

### Data integrity / security impact
- **Medium:** If deployed, would force users to make all schemas explicit
- Removes flexibility to accept any-shaped data (CSV with unknown columns)
- May cause users to bypass validation entirely

### Performance or cost impact
- Development blocked until fix is deployed
- Integration tests skipped (gap in test coverage)

## Root Cause Analysis

### Temporal Evolution of Schema Representation

**Phase 1: Original Design (before P0 fix)**
- Dynamic schemas: `None` at class level
- Detection: `if schema is None`
- Status: ❌ Broken (all schemas were None due to getattr() on config objects)

**Phase 2: After P0 Fix (commits 653dd8a, 2027906)**
- All schemas: Pydantic model instances extracted from plugin classes
- Dynamic schemas: Pydantic models with `model_fields == {}` and `extra='allow'`
- Detection: Still checking `if schema is None`
- Status: ✅ Fixed validation for explicit schemas, ❌ Broke detection of dynamic schemas

**Phase 3: Current Fix (this ticket)**
- Detection: Introspect Pydantic model structure via `_is_dynamic_schema()` helper
- Status: ✅ Both explicit AND dynamic schemas work correctly

### Why Detection Logic Wasn't Updated

**Missed during refactor:**
1. `from_plugin_instances()` refactor focused on extracting schemas from plugin classes
2. Tests at that time didn't have dynamic → specific edge cases
3. Integration test was added later, discovered the bug, and was SKIPPED with documentation
4. The skip message explicitly documented the bug for future fixing

**Root cause pattern:**
- **Representation change** (None → Pydantic instance) without **detection logic update**
- Classic refactoring oversight: Updated schema source, forgot to update schema consumer
- Defensive `getattr(..., None)` pattern masked the issue (returned instances, not None)

## Investigation Process (Systematic Debugging)

### Phase 1: Root Cause Investigation

**Evidence Gathered:**
1. ✅ Schemas are Pydantic model instances after refactor
2. ✅ Dynamic schemas have empty `model_fields` and `extra='allow'`
3. ✅ Validation logic still checks `if schema is None`
4. ✅ Runtime testing confirmed dynamic vs explicit schema characteristics
5. ✅ Identified 2 broken call sites: `_validate_edge_schemas()` line 293, `_validate_coalesce_schema_compatibility()` line 246

**Detection Signature Confirmed:**
```python
# Created schemas via schema_factory and inspected at runtime
dynamic_schema.model_fields      # {}
dynamic_schema.model_config['extra']  # 'allow'

explicit_schema.model_fields     # {'value': FieldInfo(...)}
explicit_schema.model_config['extra']  # 'forbid'
```

### Phase 2: Pattern Analysis

**Working Examples Found:**
- `SchemaConfig.is_dynamic` property used throughout plugins
- Plugins check `self._schema_config.is_dynamic` before validation
- Pattern: Config objects have explicit `is_dynamic: bool` flag

**Key Insight:**
- DAG layer doesn't have access to `SchemaConfig`, only Pydantic schema classes
- Must detect dynamic by **introspecting model structure**, not accessing config

### Phase 3: Hypothesis Formation

**Hypothesis:** Create `_is_dynamic_schema()` helper that detects dynamic schemas by:
1. Check if schema is `None` (backwards compatibility)
2. Check if `model_fields` is empty AND `model_config['extra'] == 'allow'`

**Predicted Fix Locations:**
1. Line 293-295: `_validate_edge_schemas()` - change to `if _is_dynamic_schema(producer) or _is_dynamic_schema(consumer)`
2. Line 246: `_validate_coalesce_schema_compatibility()` - change to `if not _is_dynamic_schema(schema)`

**Hypothesis Validation:** ✅ Confirmed via implementation and testing

## Implemented Fix

### Files Modified

1. **src/elspeth/core/dag.py**
   - Added `_is_dynamic_schema()` helper (lines 68-85)
   - Updated `_validate_edge_schemas()` to use helper (line 316)
   - Updated `_validate_coalesce_schema_compatibility()` to use helper (line 266)
   - Added type assertions for mypy compliance (lines 268, 320-321)

2. **tests/core/test_dag.py**
   - Added `TestDynamicSchemaDetection` class with 3 unit tests (lines 1852-1948)
   - `test_dynamic_source_to_specific_sink_should_skip_validation()`
   - `test_specific_source_to_dynamic_sink_should_skip_validation()`
   - `test_is_dynamic_schema_helper_detects_dynamic_schemas()`

3. **tests/cli/test_plugin_errors.py**
   - Unskipped `test_dynamic_schema_to_specific_schema_validation()` (removed decorator line 364)

4. **tests/integration/test_schema_validation_end_to_end.py**
   - Fixed 2 test expectations for aggregation dynamic output schemas
   - Tests now correctly expect validation to PASS (not fail)

5. **config/cicd/no_bug_hiding.yaml**
   - Added allowlist entry for `_is_dynamic_schema()` dict.get() usage
   - Updated line numbers for existing entries

### Helper Function Implementation

```python
def _is_dynamic_schema(schema: type[PluginSchema] | None) -> bool:
    """Check if a schema is dynamic (accepts any fields).

    Dynamic schemas have no defined fields and accept any extra fields.

    Args:
        schema: Schema class to check (None is treated as dynamic for backwards compat)

    Returns:
        True if schema is dynamic or None, False if explicit schema
    """
    if schema is None:
        return True  # Legacy: None = dynamic

    return (
        len(schema.model_fields) == 0 and  # No defined fields
        schema.model_config.get("extra") == "allow"  # Accepts extra fields
    )
```

**Design Rationale:**
- Backwards compatible with `None` representation
- Introspects Pydantic model structure (implementation detail, but necessary)
- Two-condition check ensures both characteristics must match
- `.get()` usage is legitimate (Pydantic framework boundary with optional dict keys)

### Validation Logic Updates

**Before:**
```python
# Line 293-295 (BROKEN)
if producer_schema is None or consumer_schema is None:
    continue
```

**After:**
```python
# Line 316 (FIXED)
if _is_dynamic_schema(producer_schema) or _is_dynamic_schema(consumer_schema):
    continue
```

**Before:**
```python
# Line 246 (BROKEN)
if schema is not None:
    incoming_schemas.append((from_node, schema))
```

**After:**
```python
# Line 266 (FIXED)
if not _is_dynamic_schema(schema):
    # Type narrowing: if not dynamic, schema is not None
    assert schema is not None
    incoming_schemas.append((from_node, schema))
```

### Test Results

```
✅ 3,279 tests pass
✅ 0 tests fail
✅ 13 tests skipped (normal)
✅ mypy type checking: clean
✅ ruff linting: clean
✅ Pre-commit hooks: all pass
```

**Integration test now passing:**
- `test_dynamic_schema_to_specific_schema_validation()` - PASS
- Verifies dynamic → specific and specific → dynamic both skip validation correctly

## Architecture Review (axiom-system-architect:architecture-critic)

**Overall Assessment:** "NEEDS REVISION" (functionally correct but introduces tech debt)

**Quality Score:** 2/5

### HIGH Severity Issues Identified

**Issue 1: Redundant Detection Mechanisms**
- Two representations of "dynamic" exist:
  1. `SchemaConfig.is_dynamic` (source of truth, explicit boolean)
  2. `_is_dynamic_schema()` (new, introspects Pydantic model)
- Knowledge duplication - definition encoded in two places
- **Recommendation:** Propagate `SchemaConfig` to DAG nodes OR use marker class pattern

**Issue 2: Pydantic Implementation Coupling**
- Fix introspects `model_fields` and `model_config.get('extra')`
- Depends on Pydantic internal structure
- Brittle if Pydantic changes how config is stored
- **Recommendation:** Use explicit marker base class (`DynamicSchemaMarker`) instead

### MEDIUM Severity Issues

**Issue 3: Legacy `None` Compatibility**
- `if schema is None: return True` path still exists
- Comment says "Legacy" but `None` still actively used by `getattr(..., None)` patterns
- Question: Is `None` legacy or intentional design?

**Issue 4: Test Expectation Inversions**
- Test names contain "incompatibility_detected" but verify validation PASSES
- Misleading for future maintainers

### Critic's Verdict

**Short-term:** Valid fix that unblocks RC-1 testing
**Long-term:** Should be flagged as tech debt with cleanup ticket
**Merge decision:** Approve for RC-1, create follow-up ticket for architectural improvement

## Code Review (axiom-python-engineering:python-code-reviewer)

**Overall Assessment:** ✅ **APPROVE - Ready to Merge**

**Quality Score:** 4/5 (Good implementation, well-tested, minor improvements possible)

### Strengths

1. ✅ **Type hints complete and accurate** - modern syntax, proper narrowing
2. ✅ **Testing quality excellent** - unit + integration coverage, real Pydantic schemas
3. ✅ **Python best practices** - DRY, PEP 8, clean logic flow
4. ✅ **Bug-hiding analysis:** `.get()` usage is LEGITIMATE (Pydantic framework boundary)
5. ✅ **CLAUDE.md compliance** - no legacy code, no defensive patterns (except legitimate `.get()`)

### Minor Issues (Non-blocking)

1. **Test naming inconsistency** - Some test names don't match behavior (should rename)
2. **Edge case documentation** - Could enhance docstring with explicit examples

### Reviewer's Verdict

**Code quality:** Production-ready with minor polish opportunities
**Blocking issues:** None
**Recommendation:** Approve for merge

## Technical Debt Tracking

### Short-term Fix (IMPLEMENTED)

✅ **What:** `_is_dynamic_schema()` helper introspects Pydantic model structure
✅ **Why:** Minimal change, unblocks RC-1 testing
✅ **Trade-off:** Creates parallel detection mechanism, couples to Pydantic internals

### Long-term Improvement (RECOMMENDED)

**Option A: Propagate SchemaConfig to DAG Nodes**
```python
# Add to NodeInfo dataclass
@dataclass
class NodeInfo:
    node_id: str
    node_type: str
    plugin_name: str
    config: dict[str, Any]
    input_schema: type[PluginSchema] | None
    output_schema: type[PluginSchema] | None
    input_schema_config: SchemaConfig | None  # NEW
    output_schema_config: SchemaConfig | None  # NEW
```

**Benefits:**
- Single source of truth (`SchemaConfig.is_dynamic`)
- No introspection needed
- Explicit rather than inferred

**Costs:**
- Larger refactor of graph construction
- Must update both `from_config()` and `from_plugin_instances()`
- Potential serialization impact (checkpointing)

**Option B: Marker Base Class Pattern**
```python
# In schema_factory.py
class DynamicSchemaMarker(PluginSchema):
    """Marker base class for dynamic schemas."""
    pass

def _create_dynamic_schema(name: str) -> type[PluginSchema]:
    return create_model(
        name,
        __base__=DynamicSchemaMarker,  # Changed base
        ...
    )

# In dag.py
def _is_dynamic_schema(schema: type[PluginSchema] | None) -> bool:
    if schema is None:
        return True
    return issubclass(schema, DynamicSchemaMarker)
```

**Benefits:**
- Explicit contract (marker interface)
- Uses `isinstance()` check (standard Python pattern)
- No Pydantic coupling

**Costs:**
- Requires modifying `schema_factory.py`
- Adds inheritance layer
- Marker pattern may be unfamiliar to some developers

### Recommendation

**For RC-1:** Ship current fix (unblocks testing)
**Post-RC-1:** Implement Option B (marker class) - cleaner, more Pythonic

## Related Bugs

- **Fixed by this ticket:** P0-2026-01-24-schema-validation-non-functional (parent bug)
- **Introduced by:** Commits 653dd8a, 2027906 (from_plugin_instances refactor)
- **Discovered during:** Plan 3 (Testing Tasks) execution
- **Blocks:** RC-1 testing with dynamic schemas

## Testing Strategy

### Unit Tests (Added)

1. **test_is_dynamic_schema_helper_detects_dynamic_schemas()**
   - Tests helper function directly
   - Covers: None case, dynamic schema, explicit schema
   - Uses real Pydantic schemas from `schema_factory`

2. **test_dynamic_source_to_specific_sink_should_skip_validation()**
   - Manually constructs graph with dynamic output, specific input
   - Verifies validation doesn't raise
   - Tests the actual fix location in `_validate_edge_schemas()`

3. **test_specific_source_to_dynamic_sink_should_skip_validation()**
   - Reverse scenario: specific output, dynamic input
   - Ensures bidirectional skip works

### Integration Tests (Unskipped)

1. **test_dynamic_schema_to_specific_schema_validation()**
   - Real YAML config through CLI
   - Two cases: dynamic→specific, specific→dynamic→specific
   - End-to-end verification

2. **test_aggregation_output_incompatibility_detected()** (expectations fixed)
   - Now correctly expects validation to PASS
   - BatchStats uses intentional dynamic output schema

### Regression Prevention

- All tests added to prevent future regressions
- Integration tests verify CLI-level behavior
- Unit tests verify helper function correctness

## Timeline

- **2026-01-24 10:00** - Bug discovered during Plan 3 test execution
- **2026-01-24 10:30** - Integration test skipped with documentation
- **2026-01-24 13:00** - Systematic debugging session (Phase 1-3)
- **2026-01-24 14:00** - Fix implemented via TDD
- **2026-01-24 14:30** - Architecture review (2/5 score)
- **2026-01-24 15:00** - Code review (4/5 score, approved)
- **2026-01-24 15:30** - All tests passing, pre-commit clean
- **2026-01-24 16:00** - This bug ticket created

## Lessons Learned

### What Went Well

1. **Systematic debugging process** - Phase 1-3 investigation identified root cause before attempting fixes
2. **Test-first approach** - Failing tests written before implementation
3. **Multi-agent review** - Architecture critic + code reviewer provided complementary perspectives
4. **Documentation during investigation** - Skip message on test documented the bug for future work

### What Could Be Improved

1. **Test coverage gap** - Dynamic schema edge cases not tested during P0 fix
2. **Refactoring discipline** - Schema representation change should have triggered detection logic review
3. **Integration test timing** - Should have been written BEFORE refactor, not after

### Process Improvements

1. **Always test representation changes** - If data structure changes, verify all consumers
2. **Property-based testing** - Consider using Hypothesis for schema compatibility testing
3. **Architectural review checkpoints** - Review should have caught parallel detection mechanisms

## Resolution

**Status:** ✅ **FIXED** (commit TBD)

**Fix verified by:**
- ✅ All 3,279 tests pass
- ✅ Integration test unskipped and passing
- ✅ Architecture critic review complete
- ✅ Code reviewer approval received
- ✅ mypy + ruff clean
- ✅ Pre-commit hooks pass

**Deployed to:** fix/rc1-bug-burndown-session-4 branch

**Follow-up tickets:**
- [ ] POST-RC1: Implement marker class pattern to eliminate Pydantic coupling
- [ ] POST-RC1: Rename tests with inverted expectations
- [ ] POST-RC1: Add property-based testing for schema compatibility

## Resolution Update

**Status:** ✅ **SUPERSEDED BY ROOT CAUSE FIX**

**Original Fix:** Phase 3 introspection (P0-2026-01-24-dynamic-schema-detection-regression)
**Root Cause Fix:** Moved validation to plugin construction (this plan)

**Why Superseded:**
The introspection fix (Phase 3) was correct but created technical debt.
Instead of fixing the debt (Phase 4 propagation plan), we fixed the
root cause (validation placement).

**See:** docs/plans/2026-01-24-fix-schema-validation-properly.md

---

## References

### Code Locations

- Helper function: `src/elspeth/core/dag.py:68-85` (REMOVED in root cause fix)
- Fix location 1: `src/elspeth/core/dag.py:316` (REMOVED in root cause fix)
- Fix location 2: `src/elspeth/core/dag.py:266` (REMOVED in root cause fix)
- Unit tests: `tests/core/test_dag.py:1852-1948`
- Integration test: `tests/cli/test_plugin_errors.py:369-415`

### Related Documentation

- CLAUDE.md: "No Bug-Hiding Patterns" policy
- CLAUDE.md: "Three-Tier Trust Model" (validates `.get()` usage)
- Plan 3: docs/plans/2026-01-24-schema-refactor-03-testing.md
- Root cause fix plan: docs/plans/2026-01-24-fix-schema-validation-properly.md

### Review Sessions

- Architecture review: Agent a21a5df (axiom-system-architect:architecture-critic)
- Code review: Agent a81f390 (axiom-python-engineering:python-code-reviewer)
- Implementation: Agent a4ffe31 (general-purpose)

---

**Ticket Status:** RESOLVED (SUPERSEDED)
**Original Resolution:** Fixed via `_is_dynamic_schema()` helper + validation logic updates
**Final Resolution:** Superseded by root cause fix (validation moved to plugin construction)
**Verification:** All tests pass, reviews approved
**Next Steps:** Original fix removed, root cause fix deployed
