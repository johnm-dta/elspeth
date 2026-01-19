# WP-16: BaseGate Test Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Migrate all test files from plugin-style `BaseGate` subclasses to config-driven `GateSettings`, completing the WP-02 gate plugin removal.

**Architecture:** WP-02 deleted concrete gate plugins (FilterGate, ThresholdGate, etc.) and WP-09 implemented config-driven gates via `GateSettings`. However, test files still create inline `BaseGate` subclasses. This plan migrates tests to use `GateSettings` with expression conditions, eliminating the deprecated pattern.

**Tech Stack:** pytest, GateSettings, expression parser

---

## Background

### The Problem

WP-02 removed plugin gates, but test files still use the old pattern:

```python
# OLD (deprecated - BaseGate subclass)
class ThresholdGate(BaseGate):
    def evaluate(self, row: Any, ctx: Any) -> GateResult:
        if row["value"] > 50:
            return GateResult(row=row, action=RoutingAction.route("high"))
        return GateResult(row=row, action=RoutingAction.continue_())
```

### The Solution

Migrate to config-driven gates:

```python
# NEW (GateSettings with expression)
gate = GateSettings(
    name="threshold",
    condition="row['value'] > 50",
    routes={"true": "high", "false": "continue"},
)
```

### Files Requiring Migration

| File | BaseGate Classes | Priority |
|------|------------------|----------|
| `tests/engine/test_orchestrator.py` | 8 | High |
| `tests/engine/test_processor.py` | 8 | High |
| `tests/engine/test_integration.py` | 7 | High |
| `tests/engine/test_orchestrator_cleanup.py` | 1 | Medium |
| `tests/plugins/test_node_id_protocol.py` | 1 | Low |
| `tests/engine/test_plugin_detection.py` | 1 | SKIP (tests isinstance) |

**Note:** `test_plugin_detection.py` tests the base class hierarchy itself and should keep its BaseGate usage.

---

## Task 1: Migrate test_orchestrator.py Gates (Part 1)

**Files:**
- Modify: `tests/engine/test_orchestrator.py`

**Step 1: Identify ThresholdGate in test_orchestrator_routes_to_named_sink**

Search for `class ThresholdGate(BaseGate)` around line 323. This gate routes values > 50 to "high" sink.

**Step 2: Replace with GateSettings**

Change from:
```python
class ThresholdGate(BaseGate):
    name = "threshold"
    input_schema = RowSchema
    output_schema = RowSchema

    def __init__(self) -> None:
        super().__init__({})

    def evaluate(self, row: Any, ctx: Any) -> GateResult:
        if row["value"] > 50:
            return GateResult(row=row, action=RoutingAction.route("high"))
        return GateResult(row=row, action=RoutingAction.continue_())
```

To:
```python
# Add to imports at top if not present
from elspeth.core.config import GateSettings

# Replace inline class with GateSettings
threshold_gate = GateSettings(
    name="threshold",
    condition="row['value'] > 50",
    routes={"true": "high", "false": "continue"},
)
```

**Step 3: Update PipelineConfig**

Change from:
```python
config = PipelineConfig(
    source=source,
    transforms=[ThresholdGate()],  # Gate in transforms list
    sinks={"default": default_sink, "high": high_sink},
)
```

To:
```python
config = PipelineConfig(
    source=source,
    transforms=[],  # No plugin gates
    sinks={"default": default_sink, "high": high_sink},
    gates=[threshold_gate],  # Config gates in gates list
)
```

**Step 4: Update graph construction if needed**

If the test manually builds the graph, ensure config gates are handled by `ExecutionGraph.from_config()` or update the manual construction.

**Step 5: Run test to verify**

Run: `pytest tests/engine/test_orchestrator.py::test_orchestrator_routes_to_named_sink -v`
Expected: PASS

**Step 6: Commit**

```bash
git add tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
refactor(test): migrate ThresholdGate to GateSettings in test_orchestrator

Replace inline BaseGate subclass with config-driven GateSettings.
Part of WP-16: BaseGate Test Cleanup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Migrate test_orchestrator.py Gates (Part 2)

**Files:**
- Modify: `tests/engine/test_orchestrator.py`

**Step 1: Identify MisroutingGate around line 844**

This gate intentionally routes to a non-existent sink for error testing.

**Step 2: Replace with GateSettings**

```python
misrouting_gate = GateSettings(
    name="misrouting_gate",
    condition="True",  # Always routes
    routes={"true": "nonexistent_sink"},  # Invalid sink
)
```

**Step 3: Update test and verify**

Run: `pytest tests/engine/test_orchestrator.py -k misrouting -v`
Expected: PASS (test should still catch the invalid sink error)

**Step 4: Commit**

```bash
git add tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
refactor(test): migrate MisroutingGate to GateSettings

Part of WP-16: BaseGate Test Cleanup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Migrate test_orchestrator.py Gates (Part 3 - RoutingGate instances)

**Files:**
- Modify: `tests/engine/test_orchestrator.py`

**Step 1: Find all RoutingGate classes**

There are ~4 RoutingGate classes at lines 1158, 3140, 3252, 3365, 3467. Each implements different routing logic.

**Step 2: Migrate each RoutingGate**

For each RoutingGate, determine the condition logic and convert to GateSettings expression.

Example pattern:
```python
# If RoutingGate routes based on "status" field:
routing_gate = GateSettings(
    name="routing_gate",
    condition="row['status']",  # Returns string route label
    routes={
        "active": "active_sink",
        "inactive": "inactive_sink",
    },
)
```

**Step 3: Run all affected tests**

Run: `pytest tests/engine/test_orchestrator.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/engine/test_orchestrator.py
git commit -m "$(cat <<'EOF'
refactor(test): migrate all RoutingGate instances to GateSettings

Converts 4 RoutingGate classes to config-driven gates.
Part of WP-16: BaseGate Test Cleanup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Migrate test_processor.py Gates

**Files:**
- Modify: `tests/engine/test_processor.py`

**Step 1: Find BaseGate classes**

There are 8 BaseGate classes including PassGate (line 483) and RouterGate (line 575).

**Step 2: Analyze each gate's logic**

- **PassGate**: Always continues (condition="True", routes={"true": "continue"})
- **RouterGate**: Routes based on field values

**Step 3: Migrate all gates to GateSettings**

Follow same pattern as Task 1-3.

**Step 4: Run tests**

Run: `pytest tests/engine/test_processor.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/engine/test_processor.py
git commit -m "$(cat <<'EOF'
refactor(test): migrate all BaseGate classes to GateSettings in test_processor

Converts 8 inline gate classes to config-driven gates.
Part of WP-16: BaseGate Test Cleanup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Migrate test_integration.py Gates

**Files:**
- Modify: `tests/engine/test_integration.py`

**Step 1: Find BaseGate classes**

There are 7 BaseGate classes including:
- EvenOddGate (line 482)
- MisroutingGate (line 627)
- SplitGate (line 983)
- ForkGate (lines 1169, 1408, 1753)

**Step 2: Migrate each gate**

For fork gates, use:
```python
fork_gate = GateSettings(
    name="fork_gate",
    condition="True",
    routes={"true": "fork"},
    fork_to=["path_a", "path_b"],
)
```

**Step 3: Run tests**

Run: `pytest tests/engine/test_integration.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/engine/test_integration.py
git commit -m "$(cat <<'EOF'
refactor(test): migrate all BaseGate classes to GateSettings in test_integration

Converts 7 inline gate classes including fork gates.
Part of WP-16: BaseGate Test Cleanup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Migrate test_orchestrator_cleanup.py Gate

**Files:**
- Modify: `tests/engine/test_orchestrator_cleanup.py`

**Step 1: Find TrackingGate (line 189)**

This gate tracks calls for cleanup verification.

**Step 2: Evaluate if tracking is needed**

If the test only needs routing behavior (not call tracking), migrate to GateSettings.
If tracking is essential, consider if the test should be rewritten or removed.

**Step 3: Migrate or update test**

**Step 4: Run tests**

Run: `pytest tests/engine/test_orchestrator_cleanup.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/engine/test_orchestrator_cleanup.py
git commit -m "$(cat <<'EOF'
refactor(test): migrate TrackingGate to GateSettings in test_orchestrator_cleanup

Part of WP-16: BaseGate Test Cleanup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Migrate test_node_id_protocol.py Gate

**Files:**
- Modify: `tests/plugins/test_node_id_protocol.py`

**Step 1: Find TestGate class**

This tests that gates implement the node_id protocol.

**Step 2: Evaluate if test is still needed**

With config-driven gates, the node_id protocol may be handled differently. If the test is obsolete, delete it. If still relevant, update to test config gate node registration.

**Step 3: Update or remove test**

**Step 4: Run tests**

Run: `pytest tests/plugins/test_node_id_protocol.py -v`
Expected: All tests PASS (or file removed)

**Step 5: Commit**

```bash
git add tests/plugins/test_node_id_protocol.py
git commit -m "$(cat <<'EOF'
refactor(test): update node_id protocol test for config gates

Part of WP-16: BaseGate Test Cleanup.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Remove BaseGate Import from Test Files

**Files:**
- Modify: All migrated test files

**Step 1: Search for unused BaseGate imports**

```bash
grep -rn "from elspeth.plugins.base import.*BaseGate" tests/
```

**Step 2: Remove unused imports**

For each file where BaseGate is no longer used, remove it from the import statement.

**Step 3: Run ruff to verify**

```bash
ruff check tests/engine/test_orchestrator.py tests/engine/test_processor.py tests/engine/test_integration.py
```

**Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(test): remove unused BaseGate imports

Cleanup after WP-16 migration to config-driven gates.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Verify No Remaining BaseGate Subclasses (except test_plugin_detection)

**Step 1: Search for remaining BaseGate subclasses**

```bash
grep -rn "class.*BaseGate" tests/ --include="*.py"
```

**Expected:** Only `tests/engine/test_plugin_detection.py` should have BaseGate classes (it tests isinstance detection).

**Step 2: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Expected:** All tests PASS

**Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: WP-16 complete - BaseGate test cleanup finished

All tests migrated from plugin-style BaseGate to config-driven GateSettings.
Only test_plugin_detection.py retains BaseGate for isinstance testing.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

- [ ] test_orchestrator.py: 0 BaseGate subclasses (was 8)
- [ ] test_processor.py: 0 BaseGate subclasses (was 8)
- [ ] test_integration.py: 0 BaseGate subclasses (was 7)
- [ ] test_orchestrator_cleanup.py: 0 BaseGate subclasses (was 1)
- [ ] test_node_id_protocol.py: 0 BaseGate subclasses (was 1)
- [ ] test_plugin_detection.py: BaseGate retained (tests isinstance)
- [ ] All tests pass
- [ ] No unused BaseGate imports
- [ ] ruff check passes

---

## Migration Pattern Reference

### Simple Threshold Gate
```python
# Before
class ThresholdGate(BaseGate):
    def evaluate(self, row, ctx):
        if row["value"] > 50:
            return GateResult(row=row, action=RoutingAction.route("high"))
        return GateResult(row=row, action=RoutingAction.continue_())

# After
threshold_gate = GateSettings(
    name="threshold",
    condition="row['value'] > 50",
    routes={"true": "high", "false": "continue"},
)
```

### String-Based Router
```python
# Before
class RouterGate(BaseGate):
    def evaluate(self, row, ctx):
        return GateResult(row=row, action=RoutingAction.route(row["status"]))

# After
router_gate = GateSettings(
    name="router",
    condition="row['status']",  # Returns string directly
    routes={"active": "active_sink", "inactive": "inactive_sink"},
)
```

### Fork Gate
```python
# Before
class ForkGate(BaseGate):
    def evaluate(self, row, ctx):
        return GateResult(row=row, action=RoutingAction.fork_to_paths(["a", "b"]))

# After
fork_gate = GateSettings(
    name="fork_gate",
    condition="True",
    routes={"true": "fork"},
    fork_to=["a", "b"],
)
```

### Pass-Through Gate
```python
# Before
class PassGate(BaseGate):
    def evaluate(self, row, ctx):
        return GateResult(row=row, action=RoutingAction.continue_())

# After
pass_gate = GateSettings(
    name="pass_gate",
    condition="True",
    routes={"true": "continue"},
)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Expression doesn't match evaluate() logic | Medium | Test fails | Carefully analyze each evaluate() method |
| Graph construction differs | Medium | Test fails | Use ExecutionGraph.from_config() where possible |
| Some gates have side effects | Low | Lost coverage | Document if any tests lose coverage |

---

## Total Estimated Time

| Task | Files | Estimated Time |
|------|-------|----------------|
| Task 1 | test_orchestrator.py (1 gate) | 20 min |
| Task 2 | test_orchestrator.py (1 gate) | 15 min |
| Task 3 | test_orchestrator.py (4 gates) | 45 min |
| Task 4 | test_processor.py (8 gates) | 1 hour |
| Task 5 | test_integration.py (7 gates) | 1 hour |
| Task 6 | test_orchestrator_cleanup.py | 20 min |
| Task 7 | test_node_id_protocol.py | 15 min |
| Task 8 | Import cleanup | 10 min |
| Task 9 | Verification | 15 min |

**Total: ~4.5 hours**
