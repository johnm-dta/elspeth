# WP-02: Gate Plugin Deletion

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Complete removal of plugin-based gates in preparation for config-driven engine-level gates (WP-09).

**Architecture:** Delete all concrete gate plugin implementations while preserving the protocol and base class infrastructure that the engine uses. Gates will become engine-level config-driven operations in WP-09.

**Tech Stack:** Python 3.12, pluggy

---

## Key Architectural Decisions

| Decision | Rationale |
|----------|-----------|
| Keep `GateProtocol` | Engine type hints and isinstance checks depend on it |
| Keep `BaseGate` | Engine isinstance checks for gate identification |
| Keep `GateResult`, `RoutingAction` | Engine uses these for routing decisions |
| Delete concrete plugins | Plugin-based gates replaced by engine-level config |
| Delete `hookimpl` | No more gate plugins to register |

---

## ⚠️ CRITICAL CONSTRAINT

**WP-02 and WP-09 MUST execute back-to-back.**

After WP-02 completes, gates will not function until WP-09 implements engine-level gates. This is intentional - the gap should be as short as possible.

**DO NOT merge WP-02 to main until WP-09 is ready to execute immediately after.**

---

## Scope

**Files to DELETE (9 files):**
```
src/elspeth/plugins/gates/filter_gate.py
src/elspeth/plugins/gates/field_match_gate.py
src/elspeth/plugins/gates/threshold_gate.py
src/elspeth/plugins/gates/hookimpl.py
src/elspeth/plugins/gates/__init__.py
tests/plugins/gates/test_filter_gate.py
tests/plugins/gates/test_field_match_gate.py
tests/plugins/gates/test_threshold_gate.py
tests/plugins/gates/__init__.py
```

**Files to MODIFY (8 files):**
- `src/elspeth/cli.py` - Remove gate imports and registry
- `src/elspeth/plugins/manager.py` - Remove builtin_gates registration
- `tests/plugins/test_base.py` - Remove gate test class
- `tests/plugins/test_protocols.py` - Remove gate conformance test
- `tests/plugins/test_integration.py` - Remove gate from integration test
- `tests/engine/test_plugin_detection.py` - Remove 2 test methods that import ThresholdGate
- `tests/plugins/test_hookimpl_registration.py` - Remove test_builtin_gates_discoverable
- `tests/integration/test_audit_integration_fixes.py` - Fix assertion expecting 3 gates

**Files to KEEP:**
- `src/elspeth/plugins/protocols.py` - `GateProtocol` (engine type contract)
- `src/elspeth/plugins/base.py` - `BaseGate` (engine isinstance checks)
- `src/elspeth/contracts/results.py` - `GateResult`, `RoutingAction` (engine routing)

**Depends on:** None
**Unlocks:** WP-09 (engine-level gates), WP-12 (utility consolidation - gate files have `_get_nested()`)

---

## Task 1: Delete Gate Plugin Source Files

**Files:**
- Delete: `src/elspeth/plugins/gates/filter_gate.py`
- Delete: `src/elspeth/plugins/gates/field_match_gate.py`
- Delete: `src/elspeth/plugins/gates/threshold_gate.py`
- Delete: `src/elspeth/plugins/gates/hookimpl.py`
- Delete: `src/elspeth/plugins/gates/__init__.py`

**Step 1: Verify files exist**

Run: `ls -la src/elspeth/plugins/gates/`

Expected: 5 Python files listed

**Step 2: Delete the files**

```bash
rm src/elspeth/plugins/gates/filter_gate.py
rm src/elspeth/plugins/gates/field_match_gate.py
rm src/elspeth/plugins/gates/threshold_gate.py
rm src/elspeth/plugins/gates/hookimpl.py
rm src/elspeth/plugins/gates/__init__.py
rmdir src/elspeth/plugins/gates
```

**Step 3: Verify deletion**

Run: `ls src/elspeth/plugins/gates/ 2>&1`

Expected: `ls: cannot access 'src/elspeth/plugins/gates/': No such file or directory`

**Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
refactor(gates): delete plugin-based gate implementations

Removes FilterGate, FieldMatchGate, ThresholdGate plugins.
Gates become engine-level config-driven operations in WP-09.

Preserves GateProtocol and BaseGate for engine infrastructure.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Delete Gate Test Files

**Files:**
- Delete: `tests/plugins/gates/test_filter_gate.py`
- Delete: `tests/plugins/gates/test_field_match_gate.py`
- Delete: `tests/plugins/gates/test_threshold_gate.py`
- Delete: `tests/plugins/gates/__init__.py`

**Step 1: Verify files exist**

Run: `ls -la tests/plugins/gates/`

Expected: 4 Python files listed

**Step 2: Delete the files**

```bash
rm tests/plugins/gates/test_filter_gate.py
rm tests/plugins/gates/test_field_match_gate.py
rm tests/plugins/gates/test_threshold_gate.py
rm tests/plugins/gates/__init__.py
rmdir tests/plugins/gates
```

**Step 3: Verify deletion**

Run: `ls tests/plugins/gates/ 2>&1`

Expected: `ls: cannot access 'tests/plugins/gates/': No such file or directory`

**Step 4: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(gates): delete plugin-based gate tests

Tests removed along with plugin implementations.
Engine-level gate tests added in WP-09.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update cli.py - Remove Gate References

**Files:**
- Modify: `src/elspeth/cli.py`

**Step 1: Read current state**

The file has:
- Line 226: `from elspeth.plugins.gates import FieldMatchGate, FilterGate, ThresholdGate`
- Lines 239-243: `GATE_PLUGINS` dict with gate registrations

**Step 2: Remove the gate import (line 226)**

Delete this line entirely:
```python
from elspeth.plugins.gates import FieldMatchGate, FilterGate, ThresholdGate
```

**Step 3: Remove the GATE_PLUGINS dict (lines 239-243)**

Delete these lines entirely:
```python
GATE_PLUGINS: dict[str, type[BaseGate]] = {
    "threshold_gate": ThresholdGate,
    "field_match_gate": FieldMatchGate,
    "filter_gate": FilterGate,
}
```

**Step 4: Remove BaseGate from imports if unused**

Check if `BaseGate` is still used elsewhere in cli.py. If only used for the GATE_PLUGINS type hint, also update line 225:

Change:
```python
from elspeth.plugins.base import BaseGate, BaseSink, BaseSource, BaseTransform
```

To:
```python
from elspeth.plugins.base import BaseSink, BaseSource, BaseTransform
```

**Step 5: Run tests to verify no import errors**

Run: `python -c "from elspeth.cli import execute_run"`

Expected: No ImportError

**Step 6: Commit**

```bash
git add src/elspeth/cli.py
git commit -m "$(cat <<'EOF'
refactor(cli): remove gate plugin references

Removes gate imports and GATE_PLUGINS registry.
Engine-level gates configured via settings in WP-09.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update manager.py - Remove Gate Registration

**Files:**
- Modify: `src/elspeth/plugins/manager.py`

**Step 1: Read current state**

The file has:
- Line 162: `from elspeth.plugins.gates.hookimpl import builtin_gates`
- Line 169: `self.register(builtin_gates)`

**Step 2: Remove the import (line 162)**

Delete this line:
```python
from elspeth.plugins.gates.hookimpl import builtin_gates
```

**Step 3: Remove the registration (line 169)**

Delete this line:
```python
self.register(builtin_gates)
```

**Step 4: Verify the method still works**

The `register_builtins()` method should now look like:
```python
def register_builtins(self) -> None:
    """Register all built-in plugin hook implementers.

    Call this once at startup to make built-in plugins discoverable.
    """
    from elspeth.plugins.sinks.hookimpl import builtin_sinks
    from elspeth.plugins.sources.hookimpl import builtin_sources
    from elspeth.plugins.transforms.hookimpl import builtin_transforms

    self.register(builtin_sources)
    self.register(builtin_transforms)
    self.register(builtin_sinks)
```

**Step 5: Run tests to verify**

Run: `python -c "from elspeth.plugins.manager import PluginManager; m = PluginManager(); m.register_builtins()"`

Expected: No ImportError

**Step 6: Commit**

```bash
git add src/elspeth/plugins/manager.py
git commit -m "$(cat <<'EOF'
refactor(manager): remove gate plugin registration

builtin_gates no longer registered - gates become engine operations.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Update test_base.py - Remove Gate Test Class

**Files:**
- Modify: `tests/plugins/test_base.py`

**Step 1: Read current state**

The file has a `TestBaseGate` class (around lines 62-93) that creates an inline ThresholdGate to test BaseGate.

**Step 2: Delete the TestBaseGate class**

Remove the entire class:
```python
class TestBaseGate:
    """Base class for gates."""

    def test_base_gate_implementation(self) -> None:
        ...
```

This is approximately lines 62-93 (the exact range may vary).

**Step 3: Run tests to verify**

Run: `pytest tests/plugins/test_base.py -v`

Expected: Remaining tests pass, no TestBaseGate tests run

**Step 4: Commit**

```bash
git add tests/plugins/test_base.py
git commit -m "$(cat <<'EOF'
test(base): remove BaseGate test class

BaseGate remains for engine isinstance checks.
Testing moved to engine-level gate tests in WP-09.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Update test_protocols.py - Remove Gate Conformance Test

**Files:**
- Modify: `tests/plugins/test_protocols.py`

**Step 1: Read current state**

The file has a `TestGateProtocol` class (lines 184-246) that tests GateProtocol conformance with an inline ThresholdGate.

**Step 2: Delete the TestGateProtocol class**

Remove the entire class:
```python
class TestGateProtocol:
    """Gate plugin protocol (routing decisions)."""

    def test_gate_implementation(self) -> None:
        ...
```

**Step 3: Run tests to verify**

Run: `pytest tests/plugins/test_protocols.py -v`

Expected: Remaining tests pass, no TestGateProtocol tests run

**Step 4: Commit**

```bash
git add tests/plugins/test_protocols.py
git commit -m "$(cat <<'EOF'
test(protocols): remove GateProtocol conformance test

GateProtocol remains for engine type contracts.
Conformance testing moved to engine-level gate tests in WP-09.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Update test_integration.py - Remove Gate Usage

**Files:**
- Modify: `tests/plugins/test_integration.py`

**Step 1: Understand the test**

The `test_full_plugin_workflow` test (lines 11-155) exercises:
`source -> transform -> gate -> sink`

It creates an inline ThresholdGate and uses gates in the workflow. Since gates are becoming engine-level, this test needs modification.

**Step 2: Simplify the test to remove gate**

The test should become:
`source -> transform -> sink`

This is a significant change. The simplest approach is to:
1. Remove the ThresholdGate class definition (lines 62-76)
2. Remove the `elspeth_get_gates` hook (lines 102-104)
3. Remove the gate instantiation and usage (lines 124, 129-130, 135, 145-148)
4. Update assertions accordingly

**Step 3: Rewrite the test**

Replace the entire `test_full_plugin_workflow` method with a gate-free version:

```python
def test_full_plugin_workflow(self) -> None:
    """Test source -> transform -> sink workflow."""
    from elspeth.plugins import (
        BaseSink,
        BaseSource,
        BaseTransform,
        PluginContext,
        PluginManager,
        PluginSchema,
        TransformResult,
        hookimpl,
    )

    # Define schemas
    class InputSchema(PluginSchema):
        value: int

    class EnrichedSchema(PluginSchema):
        value: int
        doubled: int

    class ListSource(BaseSource):
        name = "list"
        output_schema = InputSchema

        def load(self, ctx: PluginContext) -> Iterator[dict[str, Any]]:
            for value in self.config["values"]:
                yield {"value": value}

        def close(self) -> None:
            pass

    class DoubleTransform(BaseTransform):
        name = "double"
        input_schema = InputSchema
        output_schema = EnrichedSchema

        def process(
            self, row: dict[str, Any], ctx: PluginContext
        ) -> TransformResult:
            return TransformResult.success(
                {
                    "value": row["value"],
                    "doubled": row["value"] * 2,
                }
            )

        def close(self) -> None:
            pass

    class MemorySink(BaseSink):
        name = "memory"
        input_schema = EnrichedSchema
        collected: ClassVar[list[dict[str, Any]]] = []

        def write(self, row: dict[str, Any], ctx: PluginContext) -> None:
            MemorySink.collected.append(row)

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    # Register plugins
    class TestPlugin:
        @hookimpl
        def elspeth_get_source(self) -> list[type[BaseSource]]:
            return [ListSource]

        @hookimpl
        def elspeth_get_transforms(self) -> list[type[BaseTransform]]:
            return [DoubleTransform]

        @hookimpl
        def elspeth_get_sinks(self) -> list[type[BaseSink]]:
            return [MemorySink]

    manager = PluginManager()
    manager.register(TestPlugin())

    # Verify registration
    assert len(manager.get_sources()) == 1
    assert len(manager.get_transforms()) == 1
    assert len(manager.get_sinks()) == 1

    # Create instances and process
    ctx = PluginContext(run_id="test-001", config={})

    source_cls = manager.get_source_by_name("list")
    transform_cls = manager.get_transform_by_name("double")
    sink_cls = manager.get_sink_by_name("memory")

    assert source_cls is not None
    assert transform_cls is not None
    assert sink_cls is not None

    # Protocols don't define __init__ but concrete classes do
    source = source_cls({"values": [10, 50, 100]})  # type: ignore[call-arg]
    transform = transform_cls({})  # type: ignore[call-arg]
    sink = sink_cls({})  # type: ignore[call-arg]

    MemorySink.collected = []  # Reset

    for row in source.load(ctx):
        result = transform.process(row, ctx)
        assert result.status == "success"
        assert result.row is not None
        sink.write(result.row, ctx)

    # Verify all rows processed
    assert len(MemorySink.collected) == 3
    assert MemorySink.collected[0]["doubled"] == 20
    assert MemorySink.collected[1]["doubled"] == 100
    assert MemorySink.collected[2]["doubled"] == 200
```

**Step 4: Update imports at top of test**

Remove `BaseGate`, `GateResult`, `RoutingAction` if they were imported.

**Step 5: Run tests to verify**

Run: `pytest tests/plugins/test_integration.py -v`

Expected: All tests pass

**Step 6: Commit**

```bash
git add tests/plugins/test_integration.py
git commit -m "$(cat <<'EOF'
test(integration): remove gate from plugin workflow test

Simplifies test to source -> transform -> sink.
Gate routing tests moved to engine-level tests in WP-09.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Update test_plugin_detection.py - Remove Gate Import Tests

**Files:**
- Modify: `tests/engine/test_plugin_detection.py`

**Step 1: Read current state**

The file has two test methods that import from `elspeth.plugins.gates.threshold_gate`:
- `test_gate_is_base_gate()` (lines 24-29)
- `test_gate_not_transform()` (lines 123-129)

**Step 2: Delete `test_gate_is_base_gate` method**

Remove the entire method:
```python
def test_gate_is_base_gate(self) -> None:
    """Gates should be instances of BaseGate."""
    from elspeth.plugins.gates.threshold_gate import ThresholdGate

    gate = ThresholdGate({"field": "score", "threshold": 0.5})
    assert isinstance(gate, BaseGate)
```

**Step 3: Delete `test_gate_not_transform` method**

Remove the entire method:
```python
def test_gate_not_transform(self) -> None:
    """Gates should NOT be instances of BaseTransform."""
    from elspeth.plugins.gates.threshold_gate import ThresholdGate

    gate = ThresholdGate({"field": "score", "threshold": 0.5})
    # mypy knows these are incompatible hierarchies - that's what we're verifying
    assert not isinstance(gate, BaseTransform)  # type: ignore[unreachable]
```

**Step 4: Run tests to verify**

Run: `pytest tests/engine/test_plugin_detection.py -v`

Expected: Remaining tests pass

**Step 5: Commit**

```bash
git add tests/engine/test_plugin_detection.py
git commit -m "$(cat <<'EOF'
test(detection): remove gate plugin import tests

Tests that imported from deleted gate plugins removed.
Gate isinstance behavior tested in WP-09 with inline classes.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Update test_hookimpl_registration.py - Remove Gate Discovery Test

**Files:**
- Modify: `tests/plugins/test_hookimpl_registration.py`

**Step 1: Read current state**

The file has `test_builtin_gates_discoverable()` (lines 38-48) that expects gate plugins to be registered.

**Step 2: Delete the test method**

Remove the entire method:
```python
def test_builtin_gates_discoverable(self) -> None:
    """Built-in gate plugins are registered via hookimpl."""
    manager = PluginManager()
    manager.register_builtin_plugins()

    gates = manager.get_gates()
    gate_names = [g.name for g in gates]

    assert "threshold_gate" in gate_names
    assert "field_match_gate" in gate_names
    assert "filter_gate" in gate_names
```

**Step 3: Run tests to verify**

Run: `pytest tests/plugins/test_hookimpl_registration.py -v`

Expected: Remaining tests pass

**Step 4: Commit**

```bash
git add tests/plugins/test_hookimpl_registration.py
git commit -m "$(cat <<'EOF'
test(hookimpl): remove gate discovery test

No built-in gate plugins to discover after WP-02.
Gate functionality moves to engine level in WP-09.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Update test_audit_integration_fixes.py - Fix Gate Assertion

**Files:**
- Modify: `tests/integration/test_audit_integration_fixes.py`

**Step 1: Read current state**

Line 37 has:
```python
assert len(manager.get_gates()) >= 3
```

This will fail because there are no longer 3 built-in gate plugins.

**Step 2: Update the assertion**

Change line 37 from:
```python
assert len(manager.get_gates()) >= 3
```

To:
```python
assert len(manager.get_gates()) >= 0  # Gates now engine-level (WP-09)
```

Or simply delete the line if it's not essential to the test.

**Step 3: Run tests to verify**

Run: `pytest tests/integration/test_audit_integration_fixes.py -v`

Expected: Test passes

**Step 4: Commit**

```bash
git add tests/integration/test_audit_integration_fixes.py
git commit -m "$(cat <<'EOF'
test(audit): update assertion for zero built-in gates

Gate plugins removed in WP-02, engine-level gates added in WP-09.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Final Verification

**Step 1: Verify no references to deleted gate plugins**

Run:
```bash
grep -r "FilterGate\|FieldMatchGate\|ThresholdGate" src/ tests/ --include="*.py"
```

Expected: No results (or only comments/docs)

**Step 2: Verify no imports of deleted modules**

Run:
```bash
grep -r "from elspeth.plugins.gates" src/ tests/ --include="*.py"
```

Expected: No results

**Step 3: Run mypy**

Run: `mypy src/elspeth/plugins/ src/elspeth/cli.py --strict`

Expected: No errors related to missing gate imports

**Step 4: Run all plugin tests**

Run: `pytest tests/plugins/ -v`

Expected: All tests pass

**Step 5: Run full test suite**

Run: `pytest tests/ -v`

Expected: All tests pass (some engine tests may fail if they depend on gates - these will be fixed in WP-09)

**Step 6: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "$(cat <<'EOF'
chore: WP-02 cleanup - gate plugin deletion complete

All plugin-based gates removed.
Ready for WP-09: engine-level gates.

Part of WP-02: Gate Plugin Deletion

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
```

---

## Verification Checklist

- [ ] `src/elspeth/plugins/gates/` directory deleted
- [ ] `tests/plugins/gates/` directory deleted
- [ ] `cli.py` has no gate imports or GATE_PLUGINS
- [ ] `manager.py` has no builtin_gates import or registration
- [ ] `test_base.py` has no TestBaseGate class
- [ ] `test_protocols.py` has no TestGateProtocol class
- [ ] `test_integration.py` workflow test has no gates
- [ ] `test_plugin_detection.py` has no ThresholdGate import tests
- [ ] `test_hookimpl_registration.py` has no gate discovery test
- [ ] `test_audit_integration_fixes.py` assertion updated for 0 gates
- [ ] `grep` finds no FilterGate/FieldMatchGate/ThresholdGate references
- [ ] `grep` finds no `from elspeth.plugins.gates` imports
- [ ] `mypy --strict` passes on modified files
- [ ] All plugin tests pass
- [ ] All engine tests pass
- [ ] All integration tests pass
- [ ] GateProtocol still exists in protocols.py
- [ ] BaseGate still exists in base.py
- [ ] GateResult and RoutingAction still exist in results.py

---

## Files Changed Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/elspeth/plugins/gates/` | DELETE (dir) | All 5 files removed |
| `tests/plugins/gates/` | DELETE (dir) | All 4 files removed |
| `src/elspeth/cli.py` | MODIFY | Remove gate imports and registry |
| `src/elspeth/plugins/manager.py` | MODIFY | Remove builtin_gates registration |
| `tests/plugins/test_base.py` | MODIFY | Remove TestBaseGate class |
| `tests/plugins/test_protocols.py` | MODIFY | Remove TestGateProtocol class |
| `tests/plugins/test_integration.py` | MODIFY | Remove gate from workflow test |
| `tests/engine/test_plugin_detection.py` | MODIFY | Remove 2 tests importing ThresholdGate |
| `tests/plugins/test_hookimpl_registration.py` | MODIFY | Remove gate discovery test |
| `tests/integration/test_audit_integration_fixes.py` | MODIFY | Fix assertion for 0 gates |

---

## What's Preserved (Engine Infrastructure)

| File | Item | Reason |
|------|------|--------|
| `protocols.py` | `GateProtocol` | Engine type hints |
| `base.py` | `BaseGate` | Engine isinstance checks |
| `results.py` | `GateResult` | Engine routing decisions |
| `results.py` | `RoutingAction` | Engine routing actions |
| `hookspecs.py` | `elspeth_get_gates()` | Future use if needed |

---

## Next Steps After WP-02

**IMMEDIATELY execute WP-09 (Engine-Level Gates)** to restore gate functionality.

WP-09 will:
1. Create `src/elspeth/engine/expression_parser.py` - Safe expression evaluation
2. Add `GateSettings` to config
3. Implement config-driven gate evaluation in orchestrator
4. Add comprehensive tests including security fuzz tests

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Tests fail due to missing gates | Expected - engine tests using gates will fail until WP-09 |
| Other code depends on gate plugins | grep verification catches this |
| BaseGate removed accidentally | Explicit "KEEP" list and verification checklist |

---

## Rollback Plan

If issues arise:
```bash
git revert HEAD~N  # Revert last N commits from WP-02
```

The gate plugin files are in git history and can be fully restored.
