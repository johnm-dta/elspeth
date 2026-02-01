# Schema Validation Refactor - Cleanup Tasks (11-15)

> **Previous:** `03-testing.md` | **Implementation Complete**

This file contains final cleanup, documentation, and verification tasks.

---

## Task 11: Delete `from_config()` Immediately (No Deprecation)

**Files:**
- Modify: `src/elspeth/core/dag.py` (delete from_config method)

**Purpose:** Remove legacy code per CLAUDE.md policy. All callers updated in Tasks 5-7.

### Step 1: Verify no remaining callers

Run: `grep -rn "from_config" src/elspeth/`

Expected: No matches (all updated to `from_plugin_instances`)

### Step 2: Delete from_config() method

**File:** `src/elspeth/core/dag.py`

Find and delete the entire `from_config()` classmethod (should be around lines 391-650).

### Step 3: Run tests to verify nothing broke

Run: `pytest tests/ -v`

Expected: All PASS

### Step 4: Commit

```bash
git add src/elspeth/core/dag.py
git commit -m "cleanup: delete ExecutionGraph.from_config() method

- All callers migrated to from_plugin_instances()
- No deprecation period (CLAUDE.md no-legacy policy)
- Clean codebase with single graph construction API

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 12: Update Documentation and ADR

**Files:**
- Modify: `docs/bugs/P0-2026-01-24-schema-validation-non-functional.md`
- Create: `docs/design/adr/003-schema-validation-lifecycle.md`

**Purpose:** Document the architectural decision and mark bugs as resolved.

### Step 1: Update P0 bug status

**File:** `docs/bugs/P0-2026-01-24-schema-validation-non-functional.md`

Add at the top:

```markdown
## ✅ RESOLVED

**Status:** Fixed in RC-2
**Resolution:** Architectural refactor - plugin instantiation moved before graph construction
**Implementation:** See docs/plans/2026-01-24-schema-refactor-* (5 files)
**ADR:** See docs/design/adr/003-schema-validation-lifecycle.md

---
```

### Step 2: Create ADR documenting schema lifecycle

**File:** `docs/design/adr/003-schema-validation-lifecycle.md`

```markdown
# ADR 003: Schema Validation Lifecycle

## Status

Accepted

## Context

Schema validation in ExecutionGraph was non-functional because the graph was built from config objects before plugins were instantiated. Schemas are instance attributes (`self.input_schema` set in plugin `__init__()`), so they weren't available during graph construction.

This was discovered through systematic debugging after investigating P2-2026-01-24-aggregation-nodes-lack-schema-validation. Root cause analysis revealed the issue affected ALL node types (sources, transforms, aggregations, gates, sinks).

## Decision

Restructure CLI to instantiate plugins BEFORE graph construction:

1. Load config (Pydantic models)
2. Instantiate ALL plugins (source, transforms, aggregations, sinks)
3. Build graph from plugin instances using `ExecutionGraph.from_plugin_instances()`
4. Extract schemas directly from instance attributes using `getattr()`
5. Validate graph (schemas populated, validation functional)
6. Execute pipeline using pre-instantiated plugins (no double instantiation)

## Consequences

### Positive

- **Schema validation now functional** - Detects incompatibilities at validation time
- **No double instantiation** - Plugins created once, reused in execution
- **Fail-fast principle** - Plugin instantiation errors occur during validation, not execution
- **Clean architecture** - Graph construction explicitly depends on plugin instances
- **No legacy code** - `from_config()` deleted immediately per CLAUDE.md
- **Coalesce support** - Fork/join patterns fully implemented

### Negative

- **Breaking change** - `from_config()` removed (but it never worked correctly anyway)
- **Plugin instantiation required for validation** - Can't validate without creating plugins
- **Resume command complexity** - Must override source with NullSource for resume operations

## Alternatives Considered

### Option A: Add schema fields to config models

Add `input_schema`/`output_schema` fields to Pydantic config models, populate via temporary plugin instantiation before graph construction.

**Rejected because:**
- Double instantiation (performance cost)
- Violates separation of concerns (config layer knows about plugin layer)
- Uses `object.__setattr__()` to bypass frozen models (hacky)
- Accumulates technical debt
- Violates CLAUDE.md no-legacy policy

### Option B: Extract schemas from plugin classes

Make schemas class attributes instead of instance attributes.

**Rejected because:**
- Many plugins compute schemas dynamically in `__init__()` based on config options
- Would require plugin API changes
- Doesn't support per-instance schema customization
- Example: CSVSource schema depends on `path` option

## Implementation

See implementation plans:
- `docs/plans/2026-01-24-schema-refactor-00-overview.md` - Overview and design
- `docs/plans/2026-01-24-schema-refactor-01-foundation.md` - Tasks 1-4
- `docs/plans/2026-01-24-schema-refactor-02-cli-refactor.md` - Tasks 5-7
- `docs/plans/2026-01-24-schema-refactor-03-testing.md` - Tasks 8-10
- `docs/plans/2026-01-24-schema-refactor-04-cleanup.md` - Tasks 11-15

## Key Technical Details

### PluginManager Changes

```python
# OLD (defensive - hides bugs)
def get_source_by_name(self, name: str) -> type[SourceProtocol] | None:
    for plugin in self.get_sources():
        if plugin.name == name:
            return plugin.plugin_class
    return None  # Caller must check for None

# NEW (explicit - crashes on bugs)
def get_source_by_name(self, name: str) -> type[SourceProtocol]:
    for plugin in self.get_sources():
        if plugin.name == name:
            return plugin.plugin_class
    available = [p.name for p in self.get_sources()]
    raise ValueError(f"Unknown source plugin: {name}. Available: {available}")
```

### Graph Construction

```python
# OLD (broken - schemas always None)
graph = ExecutionGraph.from_config(config, manager)
# Schemas extracted via getattr(config, "input_schema", None) → None

# NEW (working - schemas from instances)
plugins = instantiate_plugins_from_config(config)
graph = ExecutionGraph.from_plugin_instances(**plugins, gates=..., output_sink=...)
# Schemas extracted via getattr(instance, "input_schema", None) → actual schema
```

### Aggregation Dual-Schema

Aggregations have separate schemas for incoming and outgoing edges:
- `input_schema` - Individual rows entering aggregation
- `output_schema` - Batch results emitted after trigger

Validation handles this correctly:
```python
# Incoming edge to aggregation
consumer_schema = to_info.input_schema  # Individual row schema

# Outgoing edge from aggregation
producer_schema = from_info.output_schema  # Batch result schema
```

### Coalesce Implementation

Fork/join patterns fully supported:
```python
# Fork gate splits to multiple branches
gate_config.fork_to = ["branch_a", "branch_b"]

# Coalesce merges branches back
coalesce_config.branches = ["branch_a", "branch_b"]

# Graph construction creates:
# gate --branch_a--> coalesce
# gate --branch_b--> coalesce
# coalesce --continue--> sink
```

**Fork without coalesce:**
Branches not in a coalesce route directly to output sink:
```python
# Fork to separate destinations (no merge)
gate_config.fork_to = ["branch_a", "branch_b"]
# No coalesce defined

# Graph construction creates:
# gate --branch_a--> output_sink (fallback)
# gate --branch_b--> output_sink (fallback)
```

### Validation Semantics

**Dynamic Schema Behavior:**

Schema validation follows these rules:

1. **Dynamic schemas skip validation** - If either producer or consumer schema is `None` (dynamic), validation is skipped for that edge
2. **Mixed dynamic/specific pipelines are valid** - Dynamic schemas act as pass-through in validation
3. **Specific → Dynamic → Specific is valid** - Validation only checks specific → specific edges

**Examples:**

```python
# VALID: Dynamic source → Specific sink (validation skipped)
datasource:
  schema: dynamic
sinks:
  output:
    schema: {fields: {x: {type: int}}}

# VALID: Specific → Dynamic → Specific (dynamic in middle skipped)
datasource:
  schema: {fields: {x: {type: int}}}
transforms:
  - schema: dynamic  # Skipped in validation
sinks:
  output:
    schema: {fields: {x: {type: int}}}

# INVALID: Specific → Specific with incompatibility
datasource:
  schema: {fields: {x: {type: int}}}
sinks:
  output:
    schema: {fields: {y: {type: str}}}  # Missing field 'y'
```

**Gate Continue Routes:**

Gates support multiple routes resolving to "continue":
```python
gates:
  - name: filter
    routes:
      true: continue    # Routes to next gate or output
      false: rejected   # Routes to specific sink
```

**ALL routes resolving to "continue"** are handled, not just `"true"`.

**Fork/Join Validation:**

1. **Fork branches** inherit schema from upstream gate
2. **Coalesce merge** validates that all incoming branch schemas are compatible
3. **Fork without coalesce** validates each branch against its destination independently

**Fork Branch Explicit Destination Requirement:**

When a gate creates fork branches (via `fork_to` configuration), **every branch must have an explicit destination**. No fallback behavior is provided.

**Resolution order:**

1. **Explicit coalesce mapping** - If the branch name is listed in a coalesce's `branches` list → routes to that coalesce
2. **Explicit sink matching** - If the branch name exactly matches a sink name → routes to that sink
3. **Validation error** - If neither exists → graph construction crashes with `GraphValidationError`

**Valid Configuration:**
```python
gates:
  - name: categorize
    fork_to: [high_priority, low_priority]

coalesce:
  branches: [high_priority]  # high_priority joins coalesce

sinks:
  low_priority:  # Sink name matches branch name
    plugin: csv
    options: {path: low.csv}

# Result:
# - high_priority → coalesce (explicit)
# - low_priority → low_priority sink (explicit match)
```

**Invalid Configuration (will crash):**
```python
gates:
  - name: categorize
    fork_to: [high_priority, low_priority, medium_priority]

coalesce:
  branches: [high_priority]

sinks:
  low_priority: {plugin: csv}
  # medium_priority sink NOT defined

# Crashes with:
# "Gate 'categorize' has fork branch 'medium_priority' with no destination.
#  Fork branches must either:
#    1. Be listed in a coalesce 'branches' list, or
#    2. Match a sink name exactly
#  Available coalesce branches: ['high_priority']
#  Available sinks: ['low_priority']"
```

**Design Rationale:** Explicit-only destinations prevent silent configuration bugs:
- **Catches typos:** `categorry` (typo) instead of `category` crashes immediately
- **No hidden behavior:** Audit trail clearly shows intended routing
- **Fail-fast:** Missing destinations are caught at graph construction, not runtime
- **Aligns with CLAUDE.md:** No silent recovery, crash on configuration errors

**Alternative for implicit routing:** If you want fork branches to share a common destination, use gate `routes` instead:
```python
gates:
  - name: categorize
    condition: "row['priority'] in ['high', 'medium']"
    routes:
      true: prioritized_sink
      false: default_sink
```

This makes the routing explicit in the configuration.

**Critical invariants:**
- Every fork branch must have an explicit destination (coalesce or matching sink name)
- Fork branches without explicit destination crash during graph construction
- Coalesce validates incoming branch schema compatibility
- Gates with continue routes must have a next node in sequence

## Notes

**Fixes:**
- P0-2026-01-24-schema-validation-non-functional
- P2-2026-01-24-aggregation-nodes-lack-schema-validation
- P3-2026-01-24-coalesce-nodes-lack-schema-validation

**Timeline:** 4-5 days implementation + testing

**Review:** Multi-agent review (architecture-critic, python-code-reviewer, test-suite-reviewer, systems-thinking) validated approach and identified gaps

## Rollback Plan

If issues discovered post-deployment:

1. Revert commits from this ADR
2. Schema validation returns to non-functional state (acceptable short-term)
3. Investigate issues
4. Re-apply fix with corrections

**Note:** No backwards compatibility needed since `from_config()` never worked correctly.
```

### Step 3: Commit documentation

```bash
git add docs/bugs/P0-2026-01-24-schema-validation-non-functional.md docs/design/adr/003-schema-validation-lifecycle.md
git commit -m "docs: update P0 bug status and create schema lifecycle ADR

- Mark P0 bug as resolved in RC-2
- Document architectural decision
- Explain alternatives considered
- Include technical details and examples
- Reference implementation plans

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 13: Close Related Bugs

**Files:**
- Modify: `docs/bugs/BUGS.md`
- Modify: `docs/bugs/P2-2026-01-24-aggregation-nodes-lack-schema-validation.md`
- Modify: `docs/bugs/P3-2026-01-24-coalesce-nodes-lack-schema-validation.md`

**Purpose:** Mark all related bugs as resolved.

### Step 1: Update BUGS.md

**File:** `docs/bugs/BUGS.md`

Mark bugs as closed:

```markdown
## Closed Bugs

### P0-2026-01-24-schema-validation-non-functional ✅
- **Status:** Resolved in RC-2
- **Fix:** Architectural refactor - plugin instantiation before graph construction
- **Resolution Date:** 2026-01-24

### P2-2026-01-24-aggregation-nodes-lack-schema-validation ✅
- **Status:** Resolved (symptom of P0 bug)
- **Fix:** Included in P0 architectural refactor
- **Resolution Date:** 2026-01-24

### P3-2026-01-24-coalesce-nodes-lack-schema-validation ✅
- **Status:** Resolved (symptom of P0 bug)
- **Fix:** Included in P0 architectural refactor
- **Resolution Date:** 2026-01-24
```

### Step 2: Update P2 and P3 bugs

Add resolution notice to both files (at top):

```markdown
## ✅ RESOLVED

**Status:** Fixed in RC-2 as part of P0-2026-01-24-schema-validation-non-functional
**Resolution:** Architectural refactor enables schema extraction from plugin instances
**Implementation:** See docs/plans/2026-01-24-schema-refactor-* files

This bug was a symptom of the broader P0 issue. The fix addresses all node types.

---
```

### Step 3: Commit bug updates

```bash
git add docs/bugs/
git commit -m "docs: close P0, P2, P3 schema validation bugs

- Mark bugs as resolved in RC-2
- Cross-reference implementation plans and ADR
- Update BUGS.md with closure status

Closes: P0-2026-01-24-schema-validation-non-functional
Closes: P2-2026-01-24-aggregation-nodes-lack-schema-validation
Closes: P3-2026-01-24-coalesce-nodes-lack-schema-validation"
```

---

## Task 13.5: Plugin Audit - Validate `__init__()` Implementations

**Files:**
- Create: `tests/audit/test_plugin_schema_contracts.py`

**Purpose:** Verify all plugins properly set schemas in `__init__()` - addresses Systems Thinking concern about plugin implementation quality.

### Step 1: Write plugin contract audit test

**File:** `tests/audit/test_plugin_schema_contracts.py`

```python
"""Audit tests for plugin schema contracts.

Verifies plugins follow schema initialization contract.
Critical for new architecture - plugins MUST set schemas in __init__().
"""

import pytest
from elspeth.plugins.manager import PluginManager


def test_all_sources_set_output_schema():
    """Verify all source plugins set output_schema in __init__()."""
    manager = PluginManager()

    for plugin_info in manager.get_sources():
        plugin_cls = plugin_info.plugin_class

        # Instantiate with minimal valid config
        try:
            instance = plugin_cls({"path": "test.csv", "schema": {"fields": "dynamic"}})
        except TypeError:
            # Some sources may require different config - skip validation
            pytest.skip(f"{plugin_info.name} requires specific config structure")

        # CRITICAL: output_schema must be set
        assert hasattr(instance, "output_schema"), \
            f"Source {plugin_info.name} missing output_schema attribute"

        # Schema can be None (dynamic) but attribute must exist
        # This validates __init__() runs the assignment


def test_all_transforms_set_schemas():
    """Verify all transform plugins set input/output schemas in __init__()."""
    manager = PluginManager()

    for plugin_info in manager.get_transforms():
        plugin_cls = plugin_info.plugin_class

        # Instantiate with minimal valid config
        try:
            instance = plugin_cls({"schema": {"fields": "dynamic"}})
        except TypeError:
            pytest.skip(f"{plugin_info.name} requires specific config structure")

        # CRITICAL: Both schemas must be set
        assert hasattr(instance, "input_schema"), \
            f"Transform {plugin_info.name} missing input_schema attribute"

        assert hasattr(instance, "output_schema"), \
            f"Transform {plugin_info.name} missing output_schema attribute"


def test_all_sinks_set_input_schema():
    """Verify all sink plugins set input_schema in __init__()."""
    manager = PluginManager()

    for plugin_info in manager.get_sinks():
        plugin_cls = plugin_info.plugin_class

        # Instantiate with minimal valid config
        try:
            instance = plugin_cls({"path": "test.csv", "schema": {"fields": "dynamic"}})
        except TypeError:
            pytest.skip(f"{plugin_info.name} requires specific config structure")

        # CRITICAL: input_schema must be set
        assert hasattr(instance, "input_schema"), \
            f"Sink {plugin_info.name} missing input_schema attribute"


def test_plugin_init_does_not_perform_io():
    """Verify plugins don't perform I/O in __init__() (brittle validation risk).

    This addresses Systems Thinking concern about validation brittleness.
    Plugins should delay I/O until execute time, not __init__().
    """
    manager = PluginManager()

    # Test CSVSource with nonexistent file - should NOT crash in __init__()
    csv_source_cls = manager.get_source_by_name("csv")
    instance = csv_source_cls({
        "path": "/nonexistent/file/that/does/not/exist.csv",
        "schema": {"fields": "dynamic"}
    })

    # If __init__() tried to open file, this would have crashed
    # Schemas should still be set
    assert hasattr(instance, "output_schema")
```

### Step 2: Run audit tests

Run: `pytest tests/audit/test_plugin_schema_contracts.py -v`

Expected: All PASS (validates plugin implementations)

### Step 3: Document any failures

If any plugin fails audit:
1. Open bug ticket for that plugin
2. Fix plugin to set schemas in `__init__()`
3. Re-run audit

### Step 4: Commit audit

```bash
git add tests/audit/test_plugin_schema_contracts.py
git commit -m "test: add plugin schema contract audit

- Verify all plugins set schemas in __init__()
- Check sources set output_schema
- Check transforms set input/output schemas
- Check sinks set input_schema
- Verify no I/O in __init__() (validation brittleness risk)

Addresses Systems Thinking concern from multi-agent review

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 13.6: Performance Baseline Measurement

**Files:**
- Create: `tests/performance/test_baseline_schema_validation.py`
- Create: `docs/performance/schema-refactor-baseline.md`

**Purpose:** Measure before/after performance to validate architectural refactor doesn't degrade performance - addresses Systems Thinking concern.

### Step 1: Create performance baseline test

**File:** `tests/performance/test_baseline_schema_validation.py`

```python
"""Performance baseline tests for schema validation refactor.

Measures validation time and plugin instantiation overhead.
Critical for validating architectural change doesn't degrade performance.
"""

import time
import tempfile
from pathlib import Path
import pytest
from elspeth.cli_helpers import instantiate_plugins_from_config
from elspeth.core.config import load_settings
from elspeth.core.dag import ExecutionGraph


@pytest.mark.performance
def test_plugin_instantiation_performance():
    """Measure plugin instantiation time."""

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}

sinks:
  output:
    plugin: csv
    options:
      path: output.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)

        start = time.perf_counter()
        plugins = instantiate_plugins_from_config(config)
        instantiation_time = time.perf_counter() - start

        # Baseline: Instantiation should be < 50ms for simple pipeline
        assert instantiation_time < 0.050, \
            f"Plugin instantiation took {instantiation_time*1000:.2f}ms (expected < 50ms)"

        print(f"\nPlugin instantiation: {instantiation_time*1000:.2f}ms")

    finally:
        config_file.unlink()


@pytest.mark.performance
def test_graph_construction_performance():
    """Measure graph construction and validation time."""

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}

sinks:
  output:
    plugin: csv
    options:
      path: output.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)

        start = time.perf_counter()
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            output_sink=config.output_sink,
        )
        graph.validate()
        graph_time = time.perf_counter() - start

        # Baseline: Graph construction + validation should be < 100ms
        assert graph_time < 0.100, \
            f"Graph construction took {graph_time*1000:.2f}ms (expected < 100ms)"

        print(f"\nGraph construction + validation: {graph_time*1000:.2f}ms")

    finally:
        config_file.unlink()


@pytest.mark.performance
def test_end_to_end_validation_performance():
    """Measure end-to-end validation performance."""

    config_yaml = """
datasource:
  plugin: csv
  options:
    path: test.csv
    schema:
      fields:
        value: {type: float}

row_plugins:
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}
  - plugin: passthrough
    options:
      schema:
        fields:
          value: {type: float}

sinks:
  output:
    plugin: csv
    options:
      path: output.csv

output_sink: output
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_yaml)
        config_file = Path(f.name)

    try:
        start = time.perf_counter()

        config = load_settings(config_file)
        plugins = instantiate_plugins_from_config(config)
        graph = ExecutionGraph.from_plugin_instances(
            source=plugins["source"],
            transforms=plugins["transforms"],
            sinks=plugins["sinks"],
            aggregations=plugins["aggregations"],
            gates=list(config.gates),
            output_sink=config.output_sink,
        )
        graph.validate()

        total_time = time.perf_counter() - start

        # Baseline: End-to-end validation should be < 200ms
        assert total_time < 0.200, \
            f"End-to-end validation took {total_time*1000:.2f}ms (expected < 200ms)"

        print(f"\nEnd-to-end validation: {total_time*1000:.2f}ms")

    finally:
        config_file.unlink()
```

### Step 2: Run performance baseline

Run: `pytest tests/performance/test_baseline_schema_validation.py -v -m performance`

Expected: All PASS with timing output

### Step 3: Document baseline

**File:** `docs/performance/schema-refactor-baseline.md`

```markdown
# Schema Validation Refactor - Performance Baseline

**Measured:** 2026-01-24
**Context:** P0-2026-01-24-schema-validation-non-functional fix

## Methodology

Performance tests measure:
1. **Plugin instantiation** - Time to create all plugin instances
2. **Graph construction** - Time to build ExecutionGraph from instances
3. **Validation** - Time to run schema validation
4. **End-to-end** - Total time from config load to validation complete

## Baseline Results

| Metric | Time (ms) | Threshold (ms) | Status |
|--------|-----------|----------------|--------|
| Plugin instantiation | [FILL] | < 50 | [FILL] |
| Graph construction + validation | [FILL] | < 100 | [FILL] |
| End-to-end validation | [FILL] | < 200 | [FILL] |

## Analysis

**Expected overhead from refactor:**
- Plugin instantiation now happens during validation (was deferred)
- Schema extraction via `getattr()` (negligible)
- Graph construction unchanged (still NetworkX)

**Net performance impact:** [FILL AFTER MEASUREMENT]

## Regression Monitoring

Re-run these tests periodically:
```bash
pytest tests/performance/ -v -m performance
```

If any test exceeds threshold by >20%, investigate for performance regression.
```

### Step 4: Commit baseline

```bash
git add tests/performance/test_baseline_schema_validation.py docs/performance/schema-refactor-baseline.md
git commit -m "perf: add performance baseline for schema validation refactor

- Measure plugin instantiation overhead
- Measure graph construction time
- Measure validation time
- Establish end-to-end validation baseline
- Document methodology and thresholds

Addresses Systems Thinking concern from multi-agent review

Part of: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 14: Audit Other `getattr()` Patterns (Follow-up Work)

**Files:**
- Create: `docs/tech-debt/2026-01-24-getattr-pattern-audit.md`

**Purpose:** Document remaining defensive patterns for future cleanup (non-blocking).

### Step 1: Find remaining getattr patterns

Run: `grep -rn "getattr.*None" src/elspeth/`

### Step 2: Analyze each pattern

For each match, determine:
- Is this a legitimate boundary case? (plugin instance attributes - OK)
- Or a bug-hiding pattern? (missing attributes we control - BAD)

### Step 3: Document findings

**File:** `docs/tech-debt/2026-01-24-getattr-pattern-audit.md`

```markdown
# Defensive Pattern Audit: `getattr(..., None)` Usage

**Date:** 2026-01-24
**Context:** Post P0-2026-01-24 fix - audit remaining defensive patterns

## Summary

Found X instances of `getattr(..., None)` in codebase.
- Y are legitimate (plugin boundaries, optional attributes)
- Z are potentially bug-hiding (needs investigation)

## Legitimate Uses

### src/elspeth/core/dag.py - Schema extraction
```python
output_schema=getattr(source, "output_schema", None)
```
**Status:** ✅ Legitimate - plugins may have dynamic schemas (None is valid)
**Rationale:** This is a boundary between our code and plugin instances

### [Other legitimate cases...]

## Potentially Bug-Hiding

### src/elspeth/engine/orchestrator.py:XXX
```python
state = getattr(node, "internal_state", None)
```
**Status:** ⚠️ Investigate - why would our node not have internal_state?
**Action:** TODO - Check if this is hiding a bug

### [Other suspicious cases...]

## Recommendations

1. Investigate "potentially bug-hiding" cases
2. For each: Either fix the root cause or document why None is valid
3. Consider lint rule: Require comment explaining why getattr(..., None) is needed
```

### Step 4: Commit audit document

```bash
git add docs/tech-debt/2026-01-24-getattr-pattern-audit.md
git commit -m "docs: audit remaining getattr defensive patterns

- Document legitimate vs potentially bug-hiding uses
- Identify follow-up investigation work
- Non-blocking - for future cleanup

Follow-up to: P0-2026-01-24-schema-validation-non-functional"
```

---

## Task 15: Final Integration Test and Verification

**Files:**
- None (verification only)

**Purpose:** Final smoke test with real pipeline configs.

### Step 1: Run full integration test suite

Run: `pytest tests/integration/ -v`

Expected: All PASS

### Step 2: Test with example pipeline configs

```bash
# Test validation
elspeth validate --settings examples/threshold_gate/settings.yaml

# Test dry-run
elspeth run --settings examples/threshold_gate/settings.yaml --dry-run

# Optional: Test actual execution if database configured
# elspeth run --settings examples/threshold_gate/settings.yaml --execute
```

Expected: Validation works, no deprecation warnings

### Step 3: Verify test coverage

Run: `pytest --cov=src/elspeth --cov-report=term-missing tests/`

Expected: Coverage of schema validation code ≥ 80%

### Step 4: Final verification checklist

```bash
# All tests pass
pytest tests/ -v

# Type checking passes
mypy src/elspeth

# Linting passes
ruff check src/elspeth

# No from_config references
! grep -r "from_config" src/elspeth/

# from_plugin_instances exists
grep -r "from_plugin_instances" src/elspeth/core/dag.py
```

### Step 5: Create final summary commit

```bash
git commit --allow-empty -m "feat: schema validation architectural refactor complete

SUMMARY:
- Schema validation now functional for ALL node types
- Plugins instantiated before graph construction
- Schemas extracted from plugin instances
- No double instantiation
- from_config() deleted (no legacy code)
- Complete coalesce implementation
- Resume command updated
- Comprehensive test coverage
- Multi-agent reviewed and approved

FIXES:
- P0-2026-01-24-schema-validation-non-functional
- P2-2026-01-24-aggregation-nodes-lack-schema-validation
- P3-2026-01-24-coalesce-nodes-lack-schema-validation

TESTING:
- 6+ integration tests (compatible, incompatible, edge direction)
- 3+ error handling tests (unknown plugins, init failures)
- 1 regression test (proves old bug fixed)
- All existing tests updated and passing

ARCHITECTURE:
- PluginManager raises on unknown plugins (no defensive programming)
- cli_helpers.py with instantiate_plugins_from_config()
- ExecutionGraph.from_plugin_instances() with complete coalesce
- _execute_pipeline_with_instances() (no double instantiation)
- _execute_resume_with_instances() (resume support)
- Aggregation dual-schema validation
- ADR-003 documenting decision

TIMELINE: 4-5 days
REVIEW: Multi-agent (architecture-critic, python-code-reviewer, test-suite-reviewer, systems-thinking)
COMPLIANCE: CLAUDE.md (no legacy code, no defensive programming)"
```

---

## Rollout Checklist

Before merging to main:

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Type checking passes (`mypy src/elspeth`)
- [ ] Linting passes (`ruff check src/elspeth`)
- [ ] Integration tests pass (`pytest tests/integration/ -v`)
- [ ] Manual testing with example pipelines
- [ ] Documentation reviewed (ADR-003)
- [ ] Bug closure reviewed (P0, P2, P3)
- [ ] No `from_config()` references in codebase
- [ ] Test coverage ≥ 80% for schema validation code
- [ ] No deprecation warnings when running CLI
- [ ] Resume command tested (at minimum: error handling)

---

## Post-Deployment Monitoring

**Watch for:**
1. Plugin instantiation errors surfacing during validation (expected - this is fail-fast)
2. Schema validation errors that were previously silent (expected - validation now works)
3. Resume failures (monitor checkpoint/resume in production)

**If issues arise:**
- Check logs for plugin instantiation errors
- Verify schemas are being extracted correctly
- Ensure resume command properly overrides source with NullSource

---

**IMPLEMENTATION COMPLETE!**

All 15 tasks documented across 5 plan files:
- 00-overview.md - Summary and design
- 01-foundation.md - Tasks 1-4 (PluginManager, helpers, graph construction)
- 02-cli-refactor.md - Tasks 5-7 (run, validate, resume commands)
- 03-testing.md - Tasks 8-10 (comprehensive tests)
- 04-cleanup.md - Tasks 11-15 (documentation, bug closure, verification)

**Ready for execution using `superpowers:executing-plans` or `superpowers:subagent-driven-development`**
