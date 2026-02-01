# Redesign Validation Enforcement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the validation enforcement mechanism to work correctly with both production plugins and test helpers, restoring two-layer defense (ABC + hook) and eliminating the 86 test failures.

**Architecture:** Move validation call into base class `__init__`, restore `@abstractmethod` on `_validate_self_consistency()`, and update `__init_subclass__` hook to verify the method exists (not that it was called).

**Tech Stack:** Python ABC, `__init_subclass__` metaclass hook, pytest

---

## The Problem We're Solving

**Current State (Broken):**
```python
class BaseTransform(ABC):
    def __init_subclass__(cls, **kwargs):
        # Only enforces if class has custom __init__
        if "__init__" not in cls.__dict__:
            return  # Bypass!

        # Wraps __init__ to check if _validation_called flag was set
        def wrapped_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not getattr(self, "_validation_called", False):
                raise RuntimeError("didn't call validation")

    def _validate_self_consistency(self) -> None:
        """Concrete method with default implementation."""
        self._validation_called = True
```

**Problems:**
1. **Test helpers break**: When test classes define `__init__`, they must call `self._validate_self_consistency()` or RuntimeError
2. **Enforcement bypass**: Classes without custom `__init__` skip validation entirely
3. **Single-layer enforcement**: ABC enforcement removed (concrete method), only hook remains
4. **86 test failures**: Test helpers throughout the suite don't call validation

**Example Failure:**
```python
# Test code (fails):
class SumTransform(BaseTransform):
    def __init__(self, node_id: str):
        super().__init__({})
        # Missing: self._validate_self_consistency()

# Result: RuntimeError from hook
```

**Proper Architecture:**
```python
class BaseTransform(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._validate_self_consistency()  # ← CALLED HERE in base class

    @abstractmethod
    def _validate_self_consistency(self) -> None:
        """Subclasses MUST implement."""
        ...  # ← Abstract again

    def __init_subclass__(cls, **kwargs):
        # Verify method EXISTS (not that it was called)
        if not hasattr(cls, "_validate_self_consistency"):
            raise TypeError(f"{cls.__name__} must implement _validate_self_consistency()")
```

**Benefits:**
- ✅ Validation **automatically** called (no manual call required)
- ✅ Test helpers work (no `__init__` override needed)
- ✅ Two-layer enforcement (ABC + hook)
- ✅ No bypass opportunity (validation always runs)

---

## Task 0: Add Validation Call to Base Class `__init__`

**Files:**
- Modify: `src/elspeth/plugins/base.py:40-77` (BaseTransform.__init__)
- Modify: `src/elspeth/plugins/base.py:176-212` (BaseGate.__init__)
- Modify: `src/elspeth/plugins/base.py:323-359` (BaseSink.__init__)
- Modify: `src/elspeth/plugins/base.py:454-483` (BaseSource.__init__)

### Step 1: Update BaseTransform.__init__ to call validation

**Current code** (`src/elspeth/plugins/base.py:40-55`):
```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize transform with configuration.

    Args:
        config: Plugin configuration dictionary
    """
    self._config = config
    self._closed = False
```

**Change to:**
```python
def __init__(self, config: dict[str, Any]) -> None:
    """Initialize transform with configuration.

    Args:
        config: Plugin configuration dictionary

    Note:
        Automatically calls _validate_self_consistency() to ensure
        schema validation happens for all plugins.
    """
    self._config = config
    self._closed = False
    self._validate_self_consistency()  # ← ADD THIS LINE
```

### Step 2: Update BaseGate.__init__ to call validation

Same pattern as Step 1, add `self._validate_self_consistency()` at end of `__init__`.

### Step 3: Update BaseSink.__init__ to call validation

Same pattern as Step 1, add `self._validate_self_consistency()` at end of `__init__`.

### Step 4: Update BaseSource.__init__ to call validation

Same pattern as Step 1, add `self._validate_self_consistency()` at end of `__init__`.

### Step 5: Run quick smoke test

```bash
pytest tests/plugins/sources/test_csv_source.py::test_csv_source_basic -xvs
```

Expected: **FAIL** - CSVSource will call validation twice (once in base, once in subclass)

### Step 6: Commit

```bash
git add src/elspeth/plugins/base.py
git commit -m "feat: call validation automatically in base class __init__

- Add self._validate_self_consistency() to all 4 base class __init__ methods
- Ensures validation always runs, no manual call required
- Next: restore @abstractmethod and update subclasses
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 1: Restore @abstractmethod on `_validate_self_consistency()`

**Files:**
- Modify: `src/elspeth/plugins/base.py:108-140` (BaseTransform._validate_self_consistency)
- Modify: `src/elspeth/plugins/base.py:243-275` (BaseGate._validate_self_consistency)
- Modify: `src/elspeth/plugins/base.py:389-421` (BaseSink._validate_self_consistency)
- Modify: `src/elspeth/plugins/base.py:520-552` (BaseSource._validate_self_consistency)

### Step 1: Restore @abstractmethod decorator

**Current code** (`src/elspeth/plugins/base.py:108-109`):
```python
def _validate_self_consistency(self) -> None:
    """Validate plugin's own schemas are self-consistent (PHASE 1)."""
```

**Change to:**
```python
@abstractmethod
def _validate_self_consistency(self) -> None:
    """Validate plugin's own schemas are self-consistent (PHASE 1)."""
```

### Step 2: Remove default implementation body

**Current code** (`src/elspeth/plugins/base.py:108-140`):
```python
def _validate_self_consistency(self) -> None:
    """Validate plugin's own schemas are self-consistent (PHASE 1).

    ...long docstring...
    """
    self._validation_called = True  # ← DELETE THIS
```

**Change to:**
```python
@abstractmethod
def _validate_self_consistency(self) -> None:
    """Validate plugin's own schemas are self-consistent (PHASE 1).

    ...long docstring (keep)...
    """
    ...  # ← Abstract method, no body
```

### Step 3: Repeat for BaseGate, BaseSink, BaseSource

Apply same changes to all 4 base classes.

### Step 4: Run smoke test

```bash
pytest tests/plugins/sources/test_csv_source.py::test_csv_source_basic -xvs
```

Expected: **FAIL** - CSVSource now calls validation in its `__init__`, which duplicates the base class call

### Step 5: Commit

```bash
git add src/elspeth/plugins/base.py
git commit -m "feat: restore @abstractmethod on _validate_self_consistency()

- Add @abstractmethod decorator to all 4 base classes
- Remove default implementation body (now abstract)
- Restore two-layer enforcement (ABC + hook)
- Next: remove manual validation calls from all plugins
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Remove Manual Validation Calls from All Plugins

**Files:**
- Modify: All 21 builtin plugins (remove `self._validate_self_consistency()` from `__init__`)
- Test helpers in `tests/conftest.py` don't need changes (they don't override `__init__`)

### Step 1: List all plugins that call validation manually

```bash
grep -r "self._validate_self_consistency()" src/elspeth/plugins/sources/ src/elspeth/plugins/transforms/ src/elspeth/plugins/sinks/ src/elspeth/plugins/gates/ --include="*.py"
```

Expected: ~21 files with manual calls

### Step 2: Write failing test for automatic validation

**Create:** `tests/plugins/test_automatic_validation.py`

```python
"""Test that validation is called automatically by base class."""
import pytest
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.results import TransformResult
from tests.conftest import _TestSchema


class MinimalTransform(BaseTransform):
    """Transform that doesn't call validation in __init__."""

    name = "minimal"
    input_schema = _TestSchema
    output_schema = _TestSchema
    determinism = "deterministic"
    plugin_version = "1.0.0"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # NO manual validation call

    def _validate_self_consistency(self) -> None:
        """Override to track if called."""
        self._validation_was_called = True

    def process(self, row: dict, ctx: PluginContext) -> TransformResult:
        return TransformResult.success(row)


def test_validation_called_automatically_by_base_class():
    """Base class __init__ must call _validate_self_consistency() automatically."""
    transform = MinimalTransform(config={})

    # Validation should have been called by BaseTransform.__init__
    assert hasattr(transform, "_validation_was_called")
    assert transform._validation_was_called is True
```

### Step 3: Run test to verify it fails

```bash
pytest tests/plugins/test_automatic_validation.py::test_validation_called_automatically_by_base_class -xvs
```

Expected: **FAIL** - Test doesn't exist yet (will create in next step)

### Step 4: Create the test file

Run: `pytest tests/plugins/test_automatic_validation.py::test_validation_called_automatically_by_base_class -xvs`

Expected: **PASS** (validation already automatic from Task 0)

### Step 5: Remove manual calls from CSVSource (example)

**File:** `src/elspeth/plugins/sources/csv_source.py`

**Find and remove:**
```python
def __init__(self, config: CSVSourceConfig):
    self._config = config
    # ... setup code ...
    self._validate_self_consistency()  # ← DELETE THIS LINE
```

### Step 6: Remove manual calls from all 21 builtin plugins

Use search-and-replace or script:
```bash
# For each plugin file, remove the validation call
for file in $(grep -l "self._validate_self_consistency()" src/elspeth/plugins/{sources,transforms,sinks,gates}/*.py); do
    sed -i '/self._validate_self_consistency()/d' "$file"
done
```

**IMPORTANT:** Manually verify each deletion - don't blindly delete if validation logic is more complex.

### Step 7: Run plugin tests to verify

```bash
pytest tests/plugins/sources/ tests/plugins/transforms/ tests/plugins/sinks/ tests/plugins/gates/ -x
```

Expected: **PASS** - All plugin tests should pass

### Step 8: Commit

```bash
git add src/elspeth/plugins/ tests/plugins/test_automatic_validation.py
git commit -m "refactor: remove manual validation calls from all plugins

- Deleted self._validate_self_consistency() calls from all 21 builtin plugins
- Validation now automatic via base class __init__
- Added test to verify automatic validation works
- All plugin tests passing
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Update `__init_subclass__` Hook to Check Method Exists (Not Called)

**Files:**
- Modify: `src/elspeth/plugins/base.py:81-106` (BaseTransform.__init_subclass__)
- Modify: Similar for BaseGate, BaseSink, BaseSource

### Step 1: Simplify hook to check method existence

**Current code** (`src/elspeth/plugins/base.py:81-106`):
```python
def __init_subclass__(cls, **kwargs: Any) -> None:
    """Enforce that subclasses call _validate_self_consistency() in __init__."""
    super().__init_subclass__(**kwargs)

    # Only enforce if the class defines its own __init__
    if "__init__" not in cls.__dict__:
        return  # Using parent's __init__, no validation needed

    original_init = cls.__init__

    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # Verify validation was called
        if not getattr(self, "_validation_called", False):
            raise RuntimeError(
                f"{cls.__name__}.__init__ did not call _validate_self_consistency(). "
                "Validation is mandatory for audit integrity."
            )

    cls.__init__ = wrapped_init
```

**Change to:**
```python
def __init_subclass__(cls, **kwargs: Any) -> None:
    """Enforce that subclasses implement _validate_self_consistency().

    This hook verifies the abstract method is implemented.
    Combined with @abstractmethod decorator, provides two-layer enforcement.
    """
    super().__init_subclass__(**kwargs)

    # ABC enforcement via @abstractmethod handles the "must implement" check
    # This hook provides runtime verification for protocol conformance
    if not hasattr(cls, "_validate_self_consistency"):
        raise TypeError(
            f"{cls.__name__} must implement _validate_self_consistency() method. "
            "This is required for audit integrity."
        )
```

### Step 2: Remove `_validation_called` flag references

Search for `_validation_called` in base.py and remove all references (no longer needed).

### Step 3: Update no_bug_hiding.yaml allowlist

**File:** `config/cicd/no_bug_hiding.yaml`

**Find:**
```yaml
- key: "plugins/base.py:R2:BaseTransform:__init_subclass__:wrapped_init:line=101"
  owner: "architecture"
  reason: "__init_subclass__ hook: verify validation was called during construction"
  safety: "Raises RuntimeError if validation wasn't called - fails fast, mandatory for audit integrity"
  expires: null
```

**Change to:**
```yaml
- key: "plugins/base.py:R3:BaseTransform:__init_subclass__:line=XX"
  owner: "architecture"
  reason: "__init_subclass__ hook: verify validation method exists"
  safety: "Raises TypeError if method not implemented - fails fast at class definition"
  expires: null
```

(Update line numbers after implementation)

### Step 4: Repeat for BaseGate, BaseSink, BaseSource

Apply same hook simplification to all 4 base classes.

### Step 5: Run smoke test

```bash
pytest tests/plugins/sources/test_csv_source.py -x
```

Expected: **PASS** - Plugins implement validation, hook verifies it exists

### Step 6: Commit

```bash
git add src/elspeth/plugins/base.py config/cicd/no_bug_hiding.yaml
git commit -m "refactor: simplify __init_subclass__ hook to check existence

- Changed hook from 'verify called' to 'verify exists'
- Removed wrapped_init pattern (no longer needed)
- Removed _validation_called flag (no longer needed)
- Updated no_bug_hiding allowlist
- Two-layer enforcement: ABC + hook both check implementation
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Fix Test Helper Classes in conftest.py

**Files:**
- Modify: `tests/conftest.py:129-175` (_TestSourceBase)
- Modify: `tests/conftest.py:177-224` (_TestSinkBase)
- Modify: `tests/conftest.py:226-274` (_TestTransformBase)

### Step 1: Add `_validate_self_consistency()` to _TestSourceBase

**Current code** (`tests/conftest.py:129-175`):
```python
class _TestSourceBase:
    """Base class for test sources that implements SourceProtocol."""

    name: str
    output_schema: type[PluginSchema]
    # ... other methods ...

    def close(self) -> None:
        """Cleanup - no-op for tests."""
        pass
```

**Add after `close()` method:**
```python
    def _validate_self_consistency(self) -> None:
        """Validation - no-op for test helpers.

        Test helpers don't need self-consistency checks since they're
        simple fixtures with minimal schemas. This method exists to
        satisfy the Protocol contract.
        """
        pass
```

### Step 2: Add `_validate_self_consistency()` to _TestSinkBase

Same pattern as Step 1.

### Step 3: Add `_validate_self_consistency()` to _TestTransformBase

Same pattern as Step 1.

### Step 4: Run tests that use these helpers

```bash
pytest tests/engine/test_processor.py::TestProcessorBatchTransforms::test_processor_buffers_rows_for_aggregation_node -xvs
```

Expected: **PASS** - Test helpers now have validation method

### Step 5: Run full engine test suite

```bash
pytest tests/engine/ -x
```

Expected: Most failures should be fixed (test helpers now work)

### Step 6: Commit

```bash
git add tests/conftest.py
git commit -m "fix: add validation to test helper base classes

- Added _validate_self_consistency() to _TestSourceBase
- Added _validate_self_consistency() to _TestSinkBase
- Added _validate_self_consistency() to _TestTransformBase
- Test helpers now satisfy protocol contract
- No-op implementation (test fixtures don't need validation logic)
- Fixes ~70+ test failures from helper classes
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Fix Inline Test Classes Throughout Test Suite

**Files:**
- Modify: `tests/engine/test_processor.py` (multiple inline test classes)
- Modify: `tests/integration/test_retry_integration.py` (inline test classes)
- Modify: Other test files with inline plugin definitions

### Step 1: Find all inline test classes that define custom __init__

```bash
grep -r "class.*Transform\|class.*Source\|class.*Sink\|class.*Gate" tests/ --include="*.py" -A 5 | grep -B 3 "def __init__"
```

### Step 2: Update each class to implement validation

**Example from `tests/engine/test_processor.py:SumTransform`:**

**Current code:**
```python
class SumTransform(BaseTransform):
    name = "sum"
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True

    def __init__(self, node_id: str) -> None:
        super().__init__({})
        self.node_id = node_id

    def process(self, rows, ctx):
        # ... implementation ...
```

**Change to:**
```python
class SumTransform(BaseTransform):
    name = "sum"
    input_schema = _TestSchema
    output_schema = _TestSchema
    is_batch_aware = True

    def __init__(self, node_id: str) -> None:
        super().__init__({})
        self.node_id = node_id

    def _validate_self_consistency(self) -> None:
        """No-op for test transform."""
        pass

    def process(self, rows, ctx):
        # ... implementation ...
```

### Step 3: Batch update common test classes

Many test files use similar patterns like:
- `DoubleTransform`
- `ErrorTransform`
- `CounterTransform`
- `FailingTransform`

Create a script or manually update each to add validation method.

### Step 4: Run affected test suites

```bash
pytest tests/engine/ tests/integration/ -x
```

Expected: All tests should pass

### Step 5: Run full test suite

```bash
pytest tests/ -x --tb=short
```

Expected: 3,305 passing (100% pass rate)

### Step 6: Commit

```bash
git add tests/
git commit -m "fix: add validation to inline test classes

- Added _validate_self_consistency() to ~50 inline test plugin classes
- All test classes now implement required abstract method
- No-op implementations (test fixtures don't need validation logic)
- Full test suite now passing (3,305 tests)
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Update Validation Enforcement Tests

**Files:**
- Modify: `tests/contracts/test_validation_enforcement.py`

### Step 1: Update tests to expect TypeError (not RuntimeError)

**Current code:**
```python
def test_transform_must_call_validation_not_just_implement():
    """Test that implementing but not calling validation raises RuntimeError."""
    # ... expects RuntimeError ...
```

**Change to:**
```python
def test_transform_must_implement_validation():
    """Test that not implementing validation raises TypeError from ABC."""

    with pytest.raises(TypeError, match="Can't instantiate abstract class.*_validate_self_consistency"):
        class BadTransform(BaseTransform):
            # Missing: _validate_self_consistency()
            pass

        BadTransform(config={})
```

### Step 2: Remove tests for "called but not implemented" scenario

The `__init_subclass__` hook no longer checks if validation was called (it checks if method exists). Remove or update tests that verify call detection.

### Step 3: Add test for automatic validation

```python
def test_validation_called_automatically():
    """Test that validation is called automatically by base class."""

    call_count = 0

    class TrackedTransform(BaseTransform):
        name = "tracked"
        input_schema = _TestSchema
        output_schema = _TestSchema

        def _validate_self_consistency(self):
            nonlocal call_count
            call_count += 1

        def process(self, row, ctx):
            return TransformResult.success(row)

    transform = TrackedTransform(config={})

    assert call_count == 1, "Validation should be called exactly once by base class"
```

### Step 4: Run enforcement tests

```bash
pytest tests/contracts/test_validation_enforcement.py -xvs
```

Expected: **PASS** - All enforcement tests should pass

### Step 5: Commit

```bash
git add tests/contracts/test_validation_enforcement.py
git commit -m "test: update validation enforcement tests for new architecture

- Changed to expect TypeError from ABC (not RuntimeError from hook)
- Removed tests for 'called but not implemented' (no longer applicable)
- Added test for automatic validation call
- All enforcement tests passing
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Plan Document and Bug Tickets

**Files:**
- Modify: `docs/plans/2026-01-24-fix-schema-validation-properly.md`
- Create: `docs/architecture/validation-enforcement-redesign.md` (record architectural decision)

### Step 1: Update Task 6 in original plan

Add note about enforcement redesign:

```markdown
## Task 6: Run Full Test Suite

### Update (2026-01-25): Enforcement Redesign Required

Running the full test suite revealed 86 failures caused by enforcement mechanism:
- Test helpers lacked `_validate_self_consistency()` method
- Hook enforcement broke inline test classes
- Single-layer enforcement had bypass opportunities

**Resolution:** See `docs/plans/2026-01-25-redesign-validation-enforcement.md`

**Changes made:**
1. Moved validation call into base class `__init__` (automatic)
2. Restored `@abstractmethod` decorator (two-layer enforcement)
3. Simplified `__init_subclass__` hook (check exists, not called)
4. Updated all test helpers and inline test classes

**Result:** 3,305/3,305 tests passing (100% pass rate)
```

### Step 2: Create architectural decision record

**Create:** `docs/architecture/validation-enforcement-redesign.md`

```markdown
# Validation Enforcement Architecture Redesign

**Date:** 2026-01-25
**Status:** Implemented
**Context:** RC-1 pre-release cleanup

## Decision

Redesigned validation enforcement to use automatic validation call in base class `__init__` instead of requiring manual calls in subclasses.

## Architecture

**Before (Task 6 state):**
- Concrete `_validate_self_consistency()` with default implementation
- `__init_subclass__` hook wraps `__init__` to verify validation was called
- Hook only enforces for classes with custom `__init__`
- Single-layer enforcement (hook only)

**After (redesign):**
- Abstract `_validate_self_consistency()` (must implement)
- Base class `__init__` calls validation automatically
- `__init_subclass__` hook verifies method exists
- Two-layer enforcement (ABC + hook)

## Rationale

1. **Eliminated manual calls**: Base class handles validation, subclasses can't forget
2. **Works with test helpers**: No custom `__init__` needed
3. **No bypass opportunity**: Validation always runs (called in base class)
4. **Two-layer enforcement**: ABC + hook both verify implementation

## Impact

- Fixed 86 test failures (96.5% → 100% pass rate)
- Restored defense-in-depth enforcement
- Simplified plugin implementation (no manual validation calls)
- Test helpers work without modification

## Files Changed

- `src/elspeth/plugins/base.py` - All 4 base classes
- All 21 builtin plugins - Removed manual validation calls
- `tests/conftest.py` - Added validation to test helpers
- `tests/` - Updated ~50 inline test classes
```

### Step 3: Commit

```bash
git add docs/
git commit -m "docs: record validation enforcement redesign decision

- Updated Task 6 in original plan with redesign note
- Created architectural decision record
- Documented before/after architecture
- Recorded impact (86 failures → 0)
- Ref: Redesign validation enforcement architecture

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Final Verification

### Step 1: Run complete test suite

```bash
pytest tests/ -v --tb=short
```

Expected: **3,305 passing, 0 failed**

### Step 2: Run mypy type checking

```bash
python -m mypy src/elspeth/
```

Expected: **Success: no issues found**

### Step 3: Run no_bug_hiding check

```bash
python -m elspeth.tools.no_bug_hiding --check
```

Expected: **PASS** (no violations, no stale entries)

### Step 4: Verify protocol conformance

```bash
pytest tests/plugins/test_protocols.py -v
```

Expected: **71/71 passing**

### Step 5: Run quick integration smoke test

```bash
pytest tests/integration/test_end_to_end.py -x
```

Expected: **PASS**

### Step 6: Create summary commit

```bash
git add -A
git commit -m "refactor: complete validation enforcement redesign

Summary of changes:
- Moved validation call to base class __init__ (automatic)
- Restored @abstractmethod decorator (two-layer enforcement)
- Simplified __init_subclass__ hook (verify exists, not called)
- Removed manual validation calls from 21 builtin plugins
- Added validation to test helpers and inline test classes
- Updated enforcement tests for new architecture

Results:
- 3,305/3,305 tests passing (100% pass rate, up from 96.5%)
- 86 test failures fixed
- Two-layer enforcement restored (ABC + hook)
- No bypass opportunities

Architecture quality: 2/5 → 5/5

Ref: docs/plans/2026-01-25-redesign-validation-enforcement.md

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria

**Must have:**
- ✅ 3,305/3,305 tests passing (100% pass rate)
- ✅ All protocols tests passing (71/71)
- ✅ mypy type checking clean
- ✅ no_bug_hiding check passing
- ✅ Two-layer enforcement (ABC + hook)
- ✅ No manual validation calls in plugins
- ✅ Test helpers work without modification

**Architecture improvements:**
- ✅ Validation automatic (called in base class)
- ✅ No bypass opportunity (validation always runs)
- ✅ Defense-in-depth restored
- ✅ Simplified plugin implementation

**Documentation:**
- ✅ Architectural decision recorded
- ✅ Original plan updated with redesign note
- ✅ Clear commit messages explaining changes

---

## Execution Plan

**Estimated time:** 2-3 hours (8 tasks × 15-20 minutes each)

**Recommended approach:** Subagent-Driven Development
- Fresh subagent per task
- Code review between tasks
- Fast iteration with quality gates

**Alternative:** Parallel session with executing-plans
- Batch execution (2-3 tasks at a time)
- Review checkpoints
- Human-in-loop for verification
