# Validation Subsystem Extraction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract plugin validation from `__init__` to a separate pre-instantiation validation subsystem, eliminating enforcement mechanism complexity while maintaining audit integrity.

**Architecture:** Create `PluginConfigValidator` that validates plugin configurations BEFORE instantiation. Plugins assume configs are valid (no validation in `__init__`). This aligns with CLAUDE.md Three-Tier Trust Model: "Plugins expect conformance - wrong types = upstream bug". Validation becomes explicit and separate from construction.

**Tech Stack:** Pydantic (config validation), existing schema_factory.py, pluggy protocols

**Key Design Decisions:**
- Validation happens ONCE before plugin creation (not on every instantiation)
- Test fixtures can bypass validation (instantiate directly for interface tests)
- Clear error messages with field-level detail (better than `__init__` exceptions)
- Enables future sandboxing (validate untrusted configs before loading)

**Deployment Strategy:** Phased rollout in 3 phases (each independently deployable):
1. Phase 1 (Discovery): Audit scope, verify assumptions, add integration tests
2. Phase 2 (Manager Integration): Add validation to PluginManager, update callsites
3. Phase 3 (Cleanup): Remove old enforcement mechanism from base classes

---

## PHASE 1: DISCOVERY AND VERIFICATION

These tasks verify assumptions, count scope accurately, and add safety tests BEFORE changing production code.

---

## Task 0.1: Audit Plugin Instantiation Sites

**Goal:** Find ALL locations where plugins are instantiated directly, so we know what needs updating.

**Files:**
- Read: `src/elspeth/engine/*.py`, `src/elspeth/cli/*.py`, `tests/**/*.py`
- Create: `docs/plans/instantiation-audit.md` (audit results)

**Step 1: Search for direct source instantiation**

Run: `grep -rn "CSVSource\|JSONSource\|NullSource\|DatabaseSource" src/elspeth/ --include="*.py" | grep -v "import" | grep -v "class " | tee instantiation-sources.txt`

Expected: Find 10-20 instantiation sites in engine/orchestrator

**Step 2: Search for direct transform instantiation**

Run: `grep -rn "Passthrough\|FieldMapper\|JSONExplode\|KeywordFilter\|Truncate\|BatchReplicate\|BatchStats" src/elspeth/ --include="*.py" | grep -v "import" | grep -v "class " | tee instantiation-transforms.txt`

Expected: Find 20-40 instantiation sites

**Step 3: Search for direct gate instantiation**

Run: `grep -rn "ThresholdGate" src/elspeth/ --include="*.py" | grep -v "import" | grep -v "class " | tee instantiation-gates.txt`

Expected: Find 5-10 instantiation sites

**Step 4: Search for direct sink instantiation**

Run: `grep -rn "CSVSink\|JSONSink\|DatabaseSink" src/elspeth/ --include="*.py" | grep -v "import" | grep -v "class " | tee instantiation-sinks.txt`

Expected: Find 10-20 instantiation sites

**Step 5: Analyze test instantiation patterns**

Run: `grep -rn "CSVSource\|JSONSource\|CSVSink" tests/ --include="*.py" | wc -l`

Expected: 200+ test instantiations (these will continue to work - direct instantiation for tests is correct)

**Step 6: Create audit report**

Create `docs/plans/instantiation-audit.md`:

```markdown
# Plugin Instantiation Audit

**Date:** 2026-01-25
**Audit Scope:** All direct plugin instantiation in src/elspeth/

## Production Code Instantiation Sites

### Sources
[Paste grep results for sources]
- Total: X sites
- Files affected: [list files]

### Transforms
[Paste grep results for transforms]
- Total: X sites
- Files affected: [list files]

### Gates
[Paste grep results for gates]
- Total: X sites
- Files affected: [list files]

### Sinks
[Paste grep results for sinks]
- Total: X sites
- Files affected: [list files]

## Summary
- **Total production instantiation sites:** X
- **Files requiring updates:** X
- **Test instantiation sites:** X (will continue to work)

## Migration Strategy
Phase 2 will update production sites to use PluginManager.create_*()
Test sites continue using direct instantiation (correct pattern)
```

**Step 7: Commit audit results**

```bash
git add docs/plans/instantiation-audit.md
git add instantiation-*.txt
git commit -m "docs: audit plugin instantiation sites for migration

Count direct instantiation in production code vs tests.
Production sites need updating to use PluginManager.create_*().
Test sites continue direct instantiation (correct pattern).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 0.2: Verify Plugin Config Models Exist

**Goal:** Confirm all builtin plugins have Pydantic config classes that validator can use.

**Files:**
- Read: `src/elspeth/plugins/sources/*.py`, `src/elspeth/plugins/transforms/*.py`, etc.
- Create: `docs/plans/config-model-audit.md`

**Step 1: List all builtin plugin files**

Run: `find src/elspeth/plugins/sources src/elspeth/plugins/transforms src/elspeth/plugins/gates src/elspeth/plugins/sinks -name "*.py" -type f | grep -v __init__ | grep -v __pycache__`

Expected: 15-20 plugin files

**Step 2: Search for config classes**

Run: `grep -rn "class.*Config.*SourceDataConfig\|TransformDataConfig\|GateDataConfig\|SinkDataConfig" src/elspeth/plugins/ --include="*.py"`

Expected: Find config class for each plugin

**Step 3: Check for from_dict method**

Run: `grep -rn "from_dict" src/elspeth/plugins/config_base.py`

Expected: Verify `from_dict()` is defined in base config classes

**Step 4: Create config model inventory**

Create `docs/plans/config-model-audit.md`:

```markdown
# Plugin Config Model Inventory

## Sources
- CSVSource → CSVSourceConfig ✓
- JSONSource → JSONSourceConfig ✓
- NullSource → NullSourceConfig ✓
- DatabaseSource → DatabaseSourceConfig ✓

## Transforms
- Passthrough → PassthroughConfig ✓
- FieldMapper → FieldMapperConfig ✓
- JSONExplode → JSONExplodeConfig ✓
- KeywordFilter → KeywordFilterConfig ✓
- Truncate → TruncateConfig ✓
- BatchReplicate → BatchReplicateConfig ✓
- BatchStats → BatchStatsConfig ✓

## Gates
- ThresholdGate → ThresholdGateConfig ✓

## Sinks
- CSVSink → CSVSinkConfig ✓
- JSONSink → JSONSinkConfig ✓
- DatabaseSink → DatabaseSinkConfig ✓

## Summary
- **Total plugins:** X
- **Plugins with config classes:** X / X (100%)
- **Config base classes have from_dict():** ✓

## Validator Implementation Notes
All plugins have compatible config classes. Validator can use:
```python
config_class.from_dict(config)  # Validates and returns instance
```
```

**Step 5: Commit inventory**

```bash
git add docs/plans/config-model-audit.md
git commit -m "docs: inventory plugin config models for validator

All builtin plugins have Pydantic config classes.
Validator can use config_class.from_dict() for validation.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 0.3: Add Pre-Implementation Integration Tests

**Goal:** Write integration tests that will FAIL until implementation is complete, serving as completion criteria.

**Files:**
- Create: `tests/plugins/test_validation_integration.py`

**Step 1: Write test for validator existence**

Create `tests/plugins/test_validation_integration.py`:

```python
"""Integration tests for validation subsystem.

These tests are written BEFORE implementation and will fail until
PluginConfigValidator and PluginManager integration are complete.

They serve as completion criteria for the migration.
"""
import pytest

from elspeth.plugins.manager import PluginManager


def test_plugin_manager_has_validator():
    """PluginManager has PluginConfigValidator instance."""
    manager = PluginManager()
    manager.register_builtin_plugins()

    # Will fail until Task 1 complete:
    assert hasattr(manager, "_validator")
    assert manager._validator is not None


def test_manager_validates_source_config_before_creation():
    """PluginManager validates source config before instantiation."""
    manager = PluginManager()
    manager.register_builtin_plugins()

    # Invalid config - missing required 'path'
    invalid_config = {
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    # Will fail until Task 4 complete:
    with pytest.raises(ValueError) as exc_info:
        manager.create_source("csv", invalid_config)

    assert "path" in str(exc_info.value)
    assert "required" in str(exc_info.value).lower()


def test_valid_config_creates_working_plugin():
    """Valid config passes validation and creates functional plugin."""
    manager = PluginManager()
    manager.register_builtin_plugins()

    valid_config = {
        "path": "/tmp/test.csv",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    # Will fail until Task 4 complete:
    source = manager.create_source("csv", valid_config)

    # Verify plugin is functional
    assert source.name == "csv"
    assert source.output_schema is not None
    assert hasattr(source, "load")


def test_validator_handles_all_builtin_sources():
    """Validator can validate configs for all builtin source types."""
    manager = PluginManager()
    manager.register_builtin_plugins()

    source_types = ["csv", "json", "null_source"]

    for source_type in source_types:
        # This will fail until Task 2 complete (validator has all types):
        # Just verify no ImportError or ValueError for unknown type
        try:
            manager.create_source(source_type, {})
        except (ValueError, TypeError):
            pass  # Config validation failure is expected
        except Exception as e:
            pytest.fail(f"Unexpected error for {source_type}: {e}")


def test_validator_provides_field_level_errors():
    """Validation errors include field name and human-readable message."""
    manager = PluginManager()
    manager.register_builtin_plugins()

    # Wrong type for skip_rows
    invalid_config = {
        "path": "/tmp/test.csv",
        "skip_rows": "not_an_int",  # Should be int
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    # Will fail until Task 1 complete:
    with pytest.raises(ValueError) as exc_info:
        manager.create_source("csv", invalid_config)

    error_msg = str(exc_info.value)
    assert "skip_rows" in error_msg  # Field name present
    # Should have human-readable message about type
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/test_validation_integration.py -xvs`

Expected: ALL tests FAIL (AttributeError, ImportError, etc.)

**Step 3: Mark tests as expected failures**

Add pytest marker to all tests:

```python
@pytest.mark.xfail(reason="Validation subsystem not yet implemented")
def test_plugin_manager_has_validator():
    ...
```

**Step 4: Run tests to verify they're marked**

Run: `pytest tests/plugins/test_validation_integration.py -v`

Expected: All tests show as XFAIL (expected failure)

**Step 5: Commit integration tests**

```bash
git add tests/plugins/test_validation_integration.py
git commit -m "test: add integration tests for validation subsystem

These tests are written BEFORE implementation (TDD).
They will fail until PluginConfigValidator and manager integration
are complete, serving as completion criteria.

Marked as xfail until implementation.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 0.4: Document Phased Rollout Strategy

**Goal:** Create deployment plan showing how to roll out changes safely in 3 phases.

**Files:**
- Create: `docs/plans/validation-migration-phases.md`

**Step 1: Write phased rollout document**

Create `docs/plans/validation-migration-phases.md`:

```markdown
# Validation Subsystem Migration - Phased Rollout

## Overview

This document describes the 3-phase rollout strategy for migrating validation
from plugin `__init__` methods to separate PluginConfigValidator subsystem.

Each phase is independently deployable and testable.

---

## Phase 1: Discovery (Tasks 0.1-0.4) ✅

**Status:** COMPLETE (you are here)

**Deliverables:**
- Instantiation audit: Know what needs updating
- Config model inventory: Verify all plugins compatible
- Integration tests: Completion criteria defined
- This document: Rollout strategy documented

**Safety:** Read-only phase, no production changes

---

## Phase 2: Manager Integration (Tasks 1-4)

**Goal:** Add PluginConfigValidator and integrate with PluginManager

**Deployment Strategy:**
1. Add validator module (Task 1)
2. Extend validator for all types (Task 2)
3. Add schema validation (Task 3)
4. Add create_* methods to manager (Task 4)

**Safety Gates:**
- Integration tests start passing (remove xfail markers)
- Old instantiation pattern STILL WORKS (backward compatible)
- Validation happens in BOTH places (manager + __init__)

**Rollback:** Revert manager changes, old enforcement still active

**Deployment Order:**
```
1. Deploy validator module (no-op, not used yet)
2. Deploy manager with create_* methods
3. Gradually migrate callsites to use manager
4. Verify dual validation doesn't break anything
```

**Success Criteria:**
- All integration tests pass
- Production can create plugins via manager.create_*()
- Old direct instantiation still works
- No test failures introduced

---

## Phase 3: Cleanup (Tasks 5-8)

**Goal:** Remove old enforcement mechanism, update test fixtures

**Deployment Strategy:**
1. Update test fixture docs (Task 5)
2. Add optional self-consistency checks (Task 6)
3. Remove enforcement from base classes (Task 7)
4. Verify all tests pass (Task 8)

**Safety Gates:**
- Manager validation proven working in Phase 2
- All production callsites migrated to manager
- Test fixtures documented to use direct instantiation

**Rollback:** Re-add enforcement mechanism if needed

**Deployment Order:**
```
1. Remove __init_subclass__ hook from base classes
2. Remove _validate_self_consistency() calls from plugins
3. Remove R2 allowlist entries
4. Deploy and verify 86 failing tests now pass
```

**Success Criteria:**
- All 3,305 tests pass (100% pass rate)
- No RuntimeError from enforcement
- Plugins are simpler (no validation in __init__)
- Test fixtures work without changes

---

## Rollback Procedures

### Phase 2 Rollback
If manager integration causes issues:
1. Revert manager.py changes
2. Production continues using direct instantiation
3. Old enforcement mechanism still works

### Phase 3 Rollback
If removing enforcement causes issues:
1. Re-add __init_subclass__ hook to base classes
2. Re-add _validate_self_consistency() calls
3. Re-add R2 allowlist entries
4. Production uses manager, tests use old pattern

---

## Risk Mitigation

### Risk: Manager methods break production
**Mitigation:** Phase 2 keeps old pattern working (backward compatible)
**Detection:** Integration tests fail
**Response:** Fix manager methods before Phase 3

### Risk: Removing enforcement breaks tests
**Mitigation:** Phase 2 proves manager validation works first
**Detection:** 86 tests fail in Phase 3
**Response:** Keep enforcement until manager proven stable

### Risk: Scope undercount (missed instantiation sites)
**Mitigation:** Task 0.1 audit finds ALL sites
**Detection:** Grep verification before Phase 3
**Response:** Update missing sites before removing enforcement

---

## Timeline

**Phase 1 (Discovery):** 4 tasks × 10 min = 40 minutes
**Phase 2 (Integration):** 4 tasks × 15 min = 60 minutes
**Phase 3 (Cleanup):** 4 tasks × 15 min = 60 minutes

**Total estimated time:** 2.5 hours (includes buffer for debugging)

---

## Success Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Test pass rate | 96.5% (3,219/3,305) | 100% (3,305/3,305) | +3.5% |
| Test failures | 86 | 0 | -86 |
| Enforcement complexity | __init_subclass__ hooks | None | Simplified |
| Validation location | Plugin __init__ | PluginManager | Centralized |
| Test fixture updates | 95+ classes | 0 | No burden |
```

**Step 2: Commit rollout strategy**

```bash
git add docs/plans/validation-migration-phases.md
git commit -m "docs: define phased rollout strategy for validation migration

3-phase deployment:
1. Discovery (read-only, no risk)
2. Manager integration (backward compatible)
3. Cleanup (enforcement removal)

Each phase independently deployable with rollback procedures.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## PHASE 2: MANAGER INTEGRATION

These tasks add validation subsystem and integrate with PluginManager. Old pattern (direct instantiation) continues to work.

---

## Task 1: Create Validation Subsystem Core

**Goal:** Create `PluginConfigValidator` class that validates plugin configurations WITHOUT instantiating plugins.

**Files:**
- Create: `src/elspeth/plugins/validation.py`
- Test: `tests/plugins/test_validation.py`

**Step 1: Write failing test for validator**

Create `tests/plugins/test_validation.py`:

```python
"""Tests for plugin configuration validation subsystem."""
from typing import Any

import pytest

from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.validation import PluginConfigValidator, ValidationError


def test_validator_accepts_valid_source_config():
    """Valid source config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/test.csv",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    errors = validator.validate_source_config("csv", config)
    assert errors == []


def test_validator_rejects_missing_required_field():
    """Missing required field returns error."""
    validator = PluginConfigValidator()

    config = {
        # Missing 'path' - required by CSVSourceConfig
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    errors = validator.validate_source_config("csv", config)
    assert len(errors) == 1
    assert "path" in errors[0].field
    assert "required" in errors[0].message.lower()


def test_validator_rejects_invalid_field_type():
    """Invalid field type returns error."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/test.csv",
        "skip_rows": "not_an_int",  # Should be int
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    errors = validator.validate_source_config("csv", config)
    assert len(errors) == 1
    assert "skip_rows" in errors[0].field
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_validation.py -xvs`
Expected: ImportError (module doesn't exist)

**Step 3: Create validation module**

Create `src/elspeth/plugins/validation.py`:

```python
"""Plugin configuration validation subsystem.

Validates plugin configurations BEFORE instantiation, providing clear error
messages and enabling test fixtures to bypass validation when needed.

Design:
- Validation is separate from plugin construction
- Returns structured errors (not exceptions) for better error messages
- Validates against Pydantic config models (CSVSourceConfig, etc.)
- Does NOT instantiate plugins (just validates config)

Usage:
    validator = PluginConfigValidator()
    errors = validator.validate_source_config("csv", config)
    if errors:
        raise ValueError(f"Invalid config: {errors}")
    source = CSVSource(config)  # Assumes config is valid
"""
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError


@dataclass
class ValidationError:
    """Structured validation error.

    Attributes:
        field: Field name that failed validation
        message: Human-readable error message
        value: The invalid value (for debugging)
    """
    field: str
    message: str
    value: Any


class PluginConfigValidator:
    """Validates plugin configurations before instantiation.

    Validates configs against Pydantic models (CSVSourceConfig, etc.)
    without actually instantiating the plugin.
    """

    def validate_source_config(
        self,
        source_type: str,
        config: dict[str, Any],
    ) -> list[ValidationError]:
        """Validate source plugin configuration.

        Args:
            source_type: Plugin type name (e.g., "csv", "json")
            config: Plugin configuration dict

        Returns:
            List of validation errors (empty if valid)
        """
        # Get config model for source type
        config_model = self._get_source_config_model(source_type)

        # Validate using Pydantic
        try:
            config_model.from_dict(config)
            return []  # Valid
        except PydanticValidationError as e:
            return self._extract_errors(e)

    def _get_source_config_model(self, source_type: str) -> type:
        """Get Pydantic config model for source type."""
        # Import here to avoid circular dependencies
        if source_type == "csv":
            from elspeth.plugins.sources.csv_source import CSVSourceConfig
            return CSVSourceConfig
        elif source_type == "json":
            from elspeth.plugins.sources.json_source import JSONSourceConfig
            return JSONSourceConfig
        elif source_type == "null_source":
            from elspeth.plugins.sources.null_source import NullSourceConfig
            return NullSourceConfig
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _extract_errors(
        self,
        pydantic_error: PydanticValidationError,
    ) -> list[ValidationError]:
        """Convert Pydantic errors to structured ValidationError list."""
        errors: list[ValidationError] = []

        for err in pydantic_error.errors():
            # Pydantic error dict has: loc, msg, type, ctx
            field_path = ".".join(str(loc) for loc in err["loc"])
            message = err["msg"]

            # Try to extract the invalid value from input
            value = err.get("input", "<unknown>")

            errors.append(ValidationError(
                field=field_path,
                message=message,
                value=value,
            ))

        return errors
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_validation.py -xvs`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/validation.py
git add tests/plugins/test_validation.py
git commit -m "feat: add plugin config validation subsystem

Create PluginConfigValidator for validating configs BEFORE instantiation.
Returns structured errors instead of raising exceptions.

- Validates against Pydantic config models (CSVSourceConfig, etc.)
- Does not instantiate plugins (separation of concerns)
- Clear field-level error messages
- Enables test fixtures to bypass validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Extend Validator for All Plugin Types

**Goal:** Add validation methods for transforms, gates, and sinks to PluginConfigValidator.

**Files:**
- Modify: `src/elspeth/plugins/validation.py`
- Test: `tests/plugins/test_validation.py`

**Step 1: Write failing tests for transform/gate/sink**

Add to `tests/plugins/test_validation.py`:

```python
def test_validator_accepts_valid_transform_config():
    """Valid transform config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "schema": {"fields": "dynamic"},
    }

    errors = validator.validate_transform_config("passthrough", config)
    assert errors == []


def test_validator_accepts_valid_gate_config():
    """Valid gate config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "field": "score",
        "threshold": 0.5,
        "schema": {"fields": "dynamic"},
    }

    errors = validator.validate_gate_config("threshold", config)
    assert errors == []


def test_validator_accepts_valid_sink_config():
    """Valid sink config passes validation."""
    validator = PluginConfigValidator()

    config = {
        "path": "/tmp/output.csv",
        "schema": {"fields": "dynamic"},
    }

    errors = validator.validate_sink_config("csv", config)
    assert errors == []
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/plugins/test_validation.py::test_validator_accepts_valid_transform_config -xvs`
Expected: AttributeError (method doesn't exist)

**Step 3: Add validation methods**

In `src/elspeth/plugins/validation.py`, add methods:

```python
def validate_transform_config(
    self,
    transform_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate transform plugin configuration."""
    config_model = self._get_transform_config_model(transform_type)

    try:
        config_model.from_dict(config)
        return []
    except PydanticValidationError as e:
        return self._extract_errors(e)


def validate_gate_config(
    self,
    gate_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate gate plugin configuration."""
    config_model = self._get_gate_config_model(gate_type)

    try:
        config_model.from_dict(config)
        return []
    except PydanticValidationError as e:
        return self._extract_errors(e)


def validate_sink_config(
    self,
    sink_type: str,
    config: dict[str, Any],
) -> list[ValidationError]:
    """Validate sink plugin configuration."""
    config_model = self._get_sink_config_model(sink_type)

    try:
        config_model.from_dict(config)
        return []
    except PydanticValidationError as e:
        return self._extract_errors(e)


def _get_transform_config_model(self, transform_type: str) -> type:
    """Get Pydantic config model for transform type."""
    if transform_type == "passthrough":
        from elspeth.plugins.transforms.passthrough import PassthroughConfig
        return PassthroughConfig
    elif transform_type == "field_mapper":
        from elspeth.plugins.transforms.field_mapper import FieldMapperConfig
        return FieldMapperConfig
    elif transform_type == "json_explode":
        from elspeth.plugins.transforms.json_explode import JSONExplodeConfig
        return JSONExplodeConfig
    elif transform_type == "keyword_filter":
        from elspeth.plugins.transforms.keyword_filter import KeywordFilterConfig
        return KeywordFilterConfig
    elif transform_type == "truncate":
        from elspeth.plugins.transforms.truncate import TruncateConfig
        return TruncateConfig
    elif transform_type == "batch_replicate":
        from elspeth.plugins.transforms.batch_replicate import BatchReplicateConfig
        return BatchReplicateConfig
    elif transform_type == "batch_stats":
        from elspeth.plugins.transforms.batch_stats import BatchStatsConfig
        return BatchStatsConfig
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def _get_gate_config_model(self, gate_type: str) -> type:
    """Get Pydantic config model for gate type."""
    if gate_type == "threshold":
        from elspeth.plugins.gates.threshold import ThresholdGateConfig
        return ThresholdGateConfig
    else:
        raise ValueError(f"Unknown gate type: {gate_type}")


def _get_sink_config_model(self, sink_type: str) -> type:
    """Get Pydantic config model for sink type."""
    if sink_type == "csv":
        from elspeth.plugins.sinks.csv_sink import CSVSinkConfig
        return CSVSinkConfig
    elif sink_type == "json":
        from elspeth.plugins.sinks.json_sink import JSONSinkConfig
        return JSONSinkConfig
    else:
        raise ValueError(f"Unknown sink type: {sink_type}")
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/plugins/test_validation.py -xvs`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/elspeth/plugins/validation.py
git add tests/plugins/test_validation.py
git commit -m "feat: add validation for transforms, gates, and sinks

Extend PluginConfigValidator to support all plugin types.

- Add validate_transform_config()
- Add validate_gate_config()
- Add validate_sink_config()
- Add config model lookups for each plugin type

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Schema Validation to Validator

**Goal:** Validate schema configurations (input_schema, output_schema) independently of plugin instantiation.

**Files:**
- Modify: `src/elspeth/plugins/validation.py`
- Test: `tests/plugins/test_validation.py`

**Step 1: Write failing test for schema validation**

Add to `tests/plugins/test_validation.py`:

```python
def test_validator_validates_schema_config():
    """Validator validates schema configuration."""
    validator = PluginConfigValidator()

    # Valid dynamic schema
    schema_config = {"fields": "dynamic"}
    errors = validator.validate_schema_config(schema_config)
    assert errors == []

    # Valid explicit schema
    schema_config = {
        "mode": "strict",
        "fields": ["id: int", "name: str"],
    }
    errors = validator.validate_schema_config(schema_config)
    assert errors == []


def test_validator_rejects_invalid_schema_mode():
    """Invalid schema mode returns error."""
    validator = PluginConfigValidator()

    schema_config = {
        "mode": "invalid_mode",  # Not "strict" or "free"
        "fields": ["id: int"],
    }

    errors = validator.validate_schema_config(schema_config)
    assert len(errors) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_validation.py::test_validator_validates_schema_config -xvs`
Expected: AttributeError (method doesn't exist)

**Step 3: Add schema validation method**

In `src/elspeth/plugins/validation.py`, add:

```python
def validate_schema_config(
    self,
    schema_config: dict[str, Any],
) -> list[ValidationError]:
    """Validate schema configuration.

    Validates SchemaConfig without creating actual Pydantic model.
    Checks field definitions, mode, and structure.

    Args:
        schema_config: Schema configuration dict

    Returns:
        List of validation errors (empty if valid)
    """
    from elspeth.contracts.schema import SchemaConfig

    try:
        SchemaConfig.from_dict(schema_config)
        return []
    except PydanticValidationError as e:
        return self._extract_errors(e)
    except Exception as e:
        # Catch other errors (e.g., ValueError from field parsing)
        return [ValidationError(
            field="schema",
            message=str(e),
            value=schema_config,
        )]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/plugins/test_validation.py::test_validator_validates_schema_config -xvs`
Expected: PASS

**Step 5: Commit**

```bash
git add src/elspeth/plugins/validation.py
git add tests/plugins/test_validation.py
git commit -m "feat: add schema configuration validation

Add validate_schema_config() to validate SchemaConfig independently.

- Validates dynamic, strict, and free mode schemas
- Checks field definitions and types
- Returns structured errors for invalid configurations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Integrate Validator into Plugin Manager

**Goal:** Update PluginManager to validate configurations before instantiating plugins.

**Files:**
- Modify: `src/elspeth/plugins/manager.py`
- Test: `tests/plugins/test_manager.py`
- Modify: `tests/plugins/test_validation_integration.py` (remove xfail markers)

**Step 1: Write failing test for manager integration**

Add to `tests/plugins/test_manager.py`:

```python
def test_manager_validates_before_instantiation(plugin_manager):
    """PluginManager validates config before creating plugin."""
    # Invalid config - missing required 'path'
    invalid_config = {
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    with pytest.raises(ValueError) as exc_info:
        plugin_manager.create_source("csv", invalid_config)

    assert "path" in str(exc_info.value)
    assert "required" in str(exc_info.value).lower()


def test_manager_creates_plugin_with_valid_config(plugin_manager):
    """PluginManager creates plugin when config is valid."""
    valid_config = {
        "path": "/tmp/test.csv",
        "schema": {"fields": "dynamic"},
        "on_validation_failure": "quarantine",
    }

    source = plugin_manager.create_source("csv", valid_config)
    assert source.name == "csv"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/plugins/test_manager.py::test_manager_validates_before_instantiation -xvs`
Expected: AttributeError (create_source method doesn't exist)

**Step 3: Add validator to PluginManager**

In `src/elspeth/plugins/manager.py`, add:

```python
from elspeth.plugins.validation import PluginConfigValidator

class PluginManager:
    """Plugin registration and instantiation manager."""

    def __init__(self) -> None:
        self._hook_manager = pluggy.PluginManager("elspeth")
        self._hook_manager.add_hookspecs(PluginHookSpecs)
        self._validator = PluginConfigValidator()  # Add validator

    def create_source(
        self,
        source_type: str,
        config: dict[str, Any],
    ) -> SourceProtocol:
        """Create source plugin with validated config.

        Args:
            source_type: Plugin type name (e.g., "csv")
            config: Plugin configuration

        Returns:
            Source plugin instance

        Raises:
            ValueError: If config is invalid
        """
        # Validate BEFORE instantiation
        errors = self._validator.validate_source_config(source_type, config)
        if errors:
            # Format errors into readable message
            error_lines = [
                f"  - {err.field}: {err.message}"
                for err in errors
            ]
            raise ValueError(
                f"Invalid config for source '{source_type}':\n"
                + "\n".join(error_lines)
            )

        # Config is valid - create plugin
        # Use existing plugin discovery mechanism
        source_class = self.get_source_by_name(source_type)
        return source_class(config)

    def create_transform(
        self,
        transform_type: str,
        config: dict[str, Any],
    ) -> TransformProtocol:
        """Create transform plugin with validated config."""
        errors = self._validator.validate_transform_config(transform_type, config)
        if errors:
            error_lines = [f"  - {err.field}: {err.message}" for err in errors]
            raise ValueError(
                f"Invalid config for transform '{transform_type}':\n"
                + "\n".join(error_lines)
            )

        transform_class = self.get_transform_by_name(transform_type)
        return transform_class(config)

    def create_gate(
        self,
        gate_type: str,
        config: dict[str, Any],
    ) -> GateProtocol:
        """Create gate plugin with validated config."""
        errors = self._validator.validate_gate_config(gate_type, config)
        if errors:
            error_lines = [f"  - {err.field}: {err.message}" for err in errors]
            raise ValueError(
                f"Invalid config for gate '{gate_type}':\n"
                + "\n".join(error_lines)
            )

        gate_class = self.get_gate_by_name(gate_type)
        return gate_class(config)

    def create_sink(
        self,
        sink_type: str,
        config: dict[str, Any],
    ) -> SinkProtocol:
        """Create sink plugin with validated config."""
        errors = self._validator.validate_sink_config(sink_type, config)
        if errors:
            error_lines = [f"  - {err.field}: {err.message}" for err in errors]
            raise ValueError(
                f"Invalid config for sink '{sink_type}':\n"
                + "\n".join(error_lines)
            )

        sink_class = self.get_sink_by_name(sink_type)
        return sink_class(config)
```

**Step 4: Remove xfail markers from integration tests**

In `tests/plugins/test_validation_integration.py`, remove `@pytest.mark.xfail` decorators from all tests.

**Step 5: Run tests to verify they pass**

Run: `pytest tests/plugins/test_manager.py tests/plugins/test_validation_integration.py -xvs`
Expected: PASS

**Step 6: Commit**

```bash
git add src/elspeth/plugins/manager.py
git add tests/plugins/test_manager.py
git add tests/plugins/test_validation_integration.py
git commit -m "feat: integrate validation into PluginManager

PluginManager now validates configs BEFORE instantiating plugins.

- Add PluginConfigValidator to manager
- Validate in create_source/transform/gate/sink methods
- Raise ValueError with field-level errors if invalid
- Integration tests now passing (xfail markers removed)

NOTE: Old pattern (direct instantiation) still works - plugins
still have validation in __init__. This provides dual validation
during migration (safety net).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## PHASE 3: CLEANUP

These tasks remove old enforcement mechanism and verify final state.

---

## Task 5: Update Test Fixtures to Document Bypass Pattern

**Goal:** Update test helper base classes in conftest.py to document that direct instantiation bypasses validation (correct for tests).

**Files:**
- Modify: `tests/conftest.py`

**Step 1: Document bypass pattern in conftest**

Add to `tests/conftest.py` after imports:

```python
"""
Test Fixture Philosophy
-----------------------
Production code validates configs BEFORE instantiation (via PluginManager).
Test fixtures bypass validation by instantiating plugins directly.

This is CORRECT because:
- Interface tests don't need config validation (testing protocol compliance)
- Test fixtures use Protocol duck typing (not base class inheritance)
- Direct instantiation is faster (no validation overhead)

Production path:
    manager = PluginManager()
    source = manager.create_source("csv", config)  # ← Validates first

Test path:
    source = MyTestSource(data)  # ← Direct instantiation, no validation
"""
```

**Step 2: Update test base class docstrings**

Update docstrings for `_TestSourceBase`, `_TestTransformBase`, `_TestSinkBase`:

```python
class _TestSourceBase:
    """Base class for test sources that implements SourceProtocol.

    NOTE: Test fixtures instantiate directly WITHOUT validation.
    This is correct - interface tests verify protocol compliance,
    not config validation. Production code validates via PluginManager.

    Provides all required Protocol attributes and lifecycle methods.
    Child classes must provide:
    - name: str
    - output_schema: type[PluginSchema]
    - load(ctx) -> Iterator[SourceRow]

    Usage:
        class MyTestSource(_TestSourceBase):
            name = "my_source"
            output_schema = MySchema

            def __init__(self, data: list[dict[str, Any]]) -> None:
                self._data = data

            def load(self, ctx: Any) -> Iterator[SourceRow]:
                yield from self.wrap_rows(self._data)
    """
```

**Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "docs: clarify test fixture validation bypass pattern

Document why test fixtures instantiate plugins directly (no validation).

- Interface tests verify protocol compliance, not config validation
- Production validates via PluginManager before instantiation
- Direct instantiation is faster and simpler for tests

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Remove Enforcement Mechanism from Base Classes

**Goal:** Remove `__init_subclass__` hook and `_validation_called` flag from all base classes.

**Files:**
- Modify: `src/elspeth/plugins/base.py` (BaseTransform, BaseGate, BaseSink, BaseSource)
- Modify: `config/cicd/no_bug_hiding.yaml` (remove R2 allowlist entries)
- Test: `tests/plugins/test_base.py`

**Step 1: Write test for no enforcement**

Add to `tests/plugins/test_base.py`:

```python
def test_plugins_instantiate_without_validation_call():
    """Plugins no longer require _validate_self_consistency() call."""
    class NoValidationTransform(BaseTransform):
        name = "no_validation"
        input_schema = TestSchema
        output_schema = TestSchema

        def __init__(self, config: dict[str, Any]) -> None:
            super().__init__(config)
            # NOT calling self._validate_self_consistency()

        def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
            return TransformResult.success(row)

    # Should instantiate without RuntimeError
    plugin = NoValidationTransform({})
    assert plugin is not None
```

**Step 2: Run test to verify it fails with current enforcement**

Run: `pytest tests/plugins/test_base.py::test_plugins_instantiate_without_validation_call -xvs`
Expected: RuntimeError from `__init_subclass__` hook

**Step 3: Remove enforcement from BaseTransform**

In `src/elspeth/plugins/base.py`, remove `__init_subclass__` and `_validate_self_consistency` methods from BaseTransform:

```python
class BaseTransform(ABC):
    """Base class for stateless row transforms."""

    name: str
    input_schema: type[PluginSchema]
    output_schema: type[PluginSchema]
    node_id: str | None = None
    determinism: Determinism = Determinism.DETERMINISTIC
    plugin_version: str = "0.0.0"
    is_batch_aware: bool = False
    creates_tokens: bool = False
    _on_error: str | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize with configuration."""
        self.config = config
        # No validation flag, no enforcement

    @abstractmethod
    def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
        """Process a single row."""
        ...

    # ... rest of methods unchanged
```

**Step 4: Remove enforcement from BaseGate, BaseSink, BaseSource**

Apply identical changes to BaseGate, BaseSink, BaseSource.

**Step 5: Remove _validate_self_consistency() calls from all builtin plugins**

For each plugin in `src/elspeth/plugins/sources/*.py`, `transforms/*.py`, `gates/*.py`, `sinks/*.py`:

Remove line: `self._validate_self_consistency()`

**Step 6: Remove no_bug_hiding.yaml allowlist entries**

In `config/cicd/no_bug_hiding.yaml`, remove R2 entries for base classes.

**Step 7: Run test to verify it passes**

Run: `pytest tests/plugins/test_base.py::test_plugins_instantiate_without_validation_call -xvs`
Expected: PASS

**Step 8: Commit**

```bash
git add src/elspeth/plugins/base.py
git add src/elspeth/plugins/sources/*.py
git add src/elspeth/plugins/transforms/*.py
git add src/elspeth/plugins/gates/*.py
git add src/elspeth/plugins/sinks/*.py
git add config/cicd/no_bug_hiding.yaml
git add tests/plugins/test_base.py
git commit -m "refactor: remove validation enforcement from base classes

Remove __init_subclass__ hook and _validate_self_consistency from all
base classes. Plugins no longer call validation in __init__.

Validation now happens BEFORE instantiation (in PluginManager), not
during construction.

- Remove enforcement from BaseTransform, BaseGate, BaseSink, BaseSource
- Remove _validate_self_consistency() calls from all builtin plugins
- Remove R2 allowlist entries from no_bug_hiding.yaml
- Test verifies plugins instantiate without validation

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Run Full Test Suite and Verify

**Goal:** Verify that all 86 previously failing tests now pass.

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short | tee test-results.txt`
Expected: 3,305 tests pass (100%)

**Step 2: Check specific failing test files**

Run:

```bash
pytest tests/engine/test_processor.py -v
pytest tests/integration/test_retry_integration.py -v
pytest tests/performance/test_baseline_schema_validation.py -v
pytest tests/system/audit_verification/test_lineage_completeness.py -v
```

Expected: All pass

**Step 3: Count test results**

Run: `grep -E "passed|failed" test-results.txt | tail -1`

Expected: `3305 passed` (100%)

**Step 4: Document results**

Create summary in commit message.

**Step 5: Commit verification**

```bash
git add test-results.txt
git commit -m "docs: verify all tests pass after validation extraction

Full test suite results:
- Before: 86 failed, 3,219 passed (96.5%)
- After: 0 failed, 3,305 passed (100%)

All failures resolved by extracting validation to separate subsystem.
Test fixtures instantiate directly (no validation), production uses
PluginManager (with validation).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Update Protocol Tests

**Goal:** Remove tests checking for `_validate_self_consistency()` method (no longer part of protocol).

**Files:**
- Modify: `tests/plugins/test_protocols.py`
- Modify: `tests/plugins/test_protocol_lifecycle.py`

**Step 1: Find validation method tests**

Run: `grep -n "_validate_self_consistency" tests/plugins/test_protocols.py tests/plugins/test_protocol_lifecycle.py`

**Step 2: Remove those tests**

Delete tests that check for validation method presence.

**Step 3: Run protocol tests**

Run: `pytest tests/plugins/test_protocols.py tests/plugins/test_protocol_lifecycle.py -xvs`
Expected: PASS

**Step 4: Commit**

```bash
git add tests/plugins/test_protocols.py
git add tests/plugins/test_protocol_lifecycle.py
git commit -m "test: remove validation method from protocol tests

Validation is no longer part of plugin protocols (moved to PluginManager).

- Remove tests checking for _validate_self_consistency() method
- Validation happens in manager.create_*() methods, not in plugins

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Success Criteria

**Must achieve:**
- ✅ All 86 failing tests now pass
- ✅ 100% test pass rate (3,305 tests)
- ✅ No enforcement mechanism in base classes
- ✅ No validation calls in plugin `__init__`
- ✅ Validation happens BEFORE instantiation (in PluginManager)
- ✅ Test fixtures can instantiate directly (no validation overhead)
- ✅ Clear error messages with field-level detail
- ✅ Integration tests passing (xfail markers removed)
- ✅ Phased rollout documented and followed

**Quality indicators:**
- ✅ No `_validation_called` flag anywhere
- ✅ No `__init_subclass__` hooks in base classes
- ✅ No R2 allowlist entries in no_bug_hiding.yaml
- ✅ Validator is separate module (clear separation of concerns)
- ✅ Plugins assume configs are valid (aligns with CLAUDE.md trust model)
- ✅ Audit trail complete (instantiation-audit.md, config-model-audit.md)

---

## Execution Notes

**TDD Discipline:**
- Every task follows: failing test → implementation → passing test → commit
- Tests written BEFORE implementation (not after)
- Each commit is green (tests pass)

**Phased Deployment:**
- Phase 1 (Discovery): Read-only, no risk
- Phase 2 (Integration): Backward compatible, dual validation
- Phase 3 (Cleanup): Safe because Phase 2 proven

**Bite-Sized Steps:**
- Each step is 2-5 minutes
- Commit after each task (not after entire plan)
- Easy to review, easy to revert

**No Over-Engineering:**
- Validator only validates configs (doesn't manage lifecycle)
- Test fixtures remain simple (direct instantiation)
- No feature flags, no backwards compatibility shims

---

## Timeline

**Phase 1 (Discovery):** 4 tasks × 10 min = 40 minutes
**Phase 2 (Integration):** 4 tasks × 15 min = 60 minutes
**Phase 3 (Cleanup):** 4 tasks × 15 min = 60 minutes

**Total estimated time:** 2.5 hours (includes buffer for debugging)

**Buffer allocation:**
- Pydantic API verification: +10 min (Task 1)
- Config model accessibility: +10 min (Task 2)
- Manager integration debugging: +15 min (Task 4)
- Test suite verification: +10 min (Task 7)

**Total with buffer:** 3 hours

---

## Alternative Approaches Considered

**Option A: Keep manual validation calls**
- Rejected: Requires updating ~95 test classes (2.4 per production plugin)
- Creates maintenance debt
- Third attempt at same fix (historical failure pattern)

**Option B: Move validation to base class `__init__`**
- Rejected: All 4 reviewers identified critical issues
- Validation runs on uninitialized state
- Violates Python idioms
- Wrong leverage point (treats symptom not cause)

**Option C: This plan - validation subsystem with phased rollout**
- Accepted: All 4 reviewers recommended this approach
- Addresses root cause (coupling between validation and construction)
- Aligns with CLAUDE.md Three-Tier Trust Model
- Enables future sandboxing (validate before loading)
- Clear separation of concerns
- Phased deployment reduces risk
