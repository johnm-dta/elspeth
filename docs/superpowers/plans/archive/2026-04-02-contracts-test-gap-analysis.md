# Contracts Test Gap Analysis — Low-Value Removal & Gap Fill

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ~150 low-value tests from `tests/unit/contracts/` and add ~20 targeted tests for untested `RunResult` and `PluginContext` recording methods.

**Architecture:** Removals target 5 anti-patterns: constructor echo, isinstance-after-construction, enum-existence, frozen-setattr, and default-value-echo. Additions cover `RunResult.__post_init__` validation and `PluginContext.record_validation_error()`/`record_transform_error()` guard clauses. All work is in the test layer — no production code changes.

**Tech Stack:** pytest, elspeth.testing factories (`make_run_result`, `make_source_context`), `unittest.mock.Mock`

---

## File Map

**Delete:**
- `tests/unit/contracts/test_export_records.py` (entire file — 24 constructor echo tests for TypedDicts with no runtime validation)

**Modify (remove low-value tests):**
- `tests/unit/contracts/test_audit.py`
- `tests/unit/contracts/test_enums.py`
- `tests/unit/contracts/test_node_state_context.py`
- `tests/unit/contracts/test_field_contract.py`
- `tests/unit/contracts/test_routing.py`
- `tests/unit/contracts/test_transform_contract.py`
- `tests/unit/contracts/test_errors.py`
- `tests/unit/contracts/test_telemetry_config.py`
- `tests/unit/contracts/test_new_errors.py`
- `tests/unit/contracts/test_checkpoint.py`
- `tests/unit/contracts/test_engine_contracts.py`
- `tests/unit/contracts/test_batch_checkpoint.py`
- `tests/unit/contracts/test_diversion.py`
- `tests/unit/contracts/test_diverted_outcome.py`
- `tests/unit/contracts/test_events.py`
- `tests/unit/contracts/test_coalesce_enums.py`
- `tests/unit/contracts/test_coalesce_metadata.py`
- `tests/unit/contracts/test_probes.py`
- `tests/unit/contracts/test_results.py`
- `tests/unit/contracts/test_source_row_contract.py`
- `tests/unit/contracts/test_contract_violations.py`
- `tests/unit/contracts/test_config.py`

**Create:**
- `tests/unit/contracts/test_run_result.py`
- `tests/unit/contracts/test_plugin_context_recording.py`

---

### Task 1: Baseline — Record Current Test Count

- [ ] **Step 1: Run contracts unit tests and record baseline**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ --co -q 2>&1 | tail -3`

Expected: `2204 tests collected` (or similar — record the exact number)

- [ ] **Step 2: Run all contracts tests to confirm green baseline**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -x -q`

Expected: All pass, 0 failures.

---

### Task 2: Delete `test_export_records.py`

**Files:**
- Delete: `tests/unit/contracts/test_export_records.py`

All 24 tests are pure TypedDict constructor echo tests — pass values in, assert same values come out. TypedDicts have no `__post_init__`, no validation, no runtime behavior. The file header itself states "Full correctness is verified by exporter integration tests."

- [ ] **Step 1: Delete the file**

```bash
rm tests/unit/contracts/test_export_records.py
```

- [ ] **Step 2: Run tests to confirm nothing breaks**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -x -q`

Expected: All pass. Test count drops by 24.

- [ ] **Step 3: Commit**

```bash
git add -u tests/unit/contracts/test_export_records.py
git commit -m "test: remove 24 low-value TypedDict echo tests from test_export_records.py"
```

---

### Task 3: Trim `test_audit.py` — Remove Enum Existence & Constructor Echo Tests

**Files:**
- Modify: `tests/unit/contracts/test_audit.py`

Remove these specific test functions. Each is either an enum-existence check, a constructor echo, or a frozen-setattr test:

**Enum existence / enum value checks (remove all):**
- `test_run_status_must_be_enum`
- `test_node_type_is_enum`
- `test_determinism_is_enum`
- `test_default_mode_is_enum`
- `test_row_outcome_all_values_known`
- `test_node_state_status_all_values_known`
- `test_run_status_all_values_known`
- `test_node_type_all_values_known`
- `test_determinism_all_values_known`
- `test_call_type_all_values_known`
- `test_call_status_all_values_known`
- `test_batch_status_all_values_known`
- `test_routing_mode_all_values_known`
- `test_export_status_all_values_known`

**Constructor echo tests (remove all):**
- `test_create_run_with_required_fields`
- `test_create_node_with_enum_fields`
- `test_create_edge_with_routing_mode`
- `test_create_row`
- `test_row_with_payload_ref`
- `test_create_token`
- `test_token_with_fork_fields`
- `test_create_token_parent`
- `test_multi_parent_ordinal`
- `test_run_with_export_status`

**Frozen setattr (parametrized — remove entire test):**
- `test_frozen_dataclass_rejects_mutation`

- [ ] **Step 1: Remove all listed test functions from the file**

Delete each listed function (and its class if the class becomes empty after removal). Preserve all remaining tests — especially `test_run_requires_*`, `test_*_rejects_*`, serialization round-trips, and regression tests.

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_audit.py -x -q`

Expected: All remaining tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_audit.py
git commit -m "test: remove 25 low-value enum/echo/frozen tests from test_audit.py"
```

---

### Task 4: Trim `test_enums.py` — Remove Enum Existence Checks

**Files:**
- Modify: `tests/unit/contracts/test_enums.py`

Remove these test functions:
- `test_has_all_required_values`
- `test_no_unknown_value`
- `test_string_values_match_architecture`
- `test_is_str_enum`
- `test_has_all_terminal_states`
- `test_row_outcome_expanded_exists`
- `test_row_outcome_buffered_exists`
- `test_all_outcomes_have_is_terminal`
- `test_routing_mode_values`
- `test_routing_mode_divert_is_str`
- `test_trigger_type_exists`
- `test_trigger_type_values`
- `test_trigger_type_is_str_enum`

Keep: `test_terminal_mappings`, all coercion tests, all `test_*_invalid_*` tests.

- [ ] **Step 1: Remove all listed test functions**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_enums.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_enums.py
git commit -m "test: remove 13 low-value enum existence checks from test_enums.py"
```

---

### Task 5: Trim `test_node_state_context.py` — Remove Frozen & Echo Tests

**Files:**
- Modify: `tests/unit/contracts/test_node_state_context.py`

Remove these test functions (they exist inside test classes — remove from the class, delete class if empty):
- `TestPoolConfigSnapshot::test_to_dict`
- `TestPoolConfigSnapshot::test_frozen`
- `TestPoolStatsSnapshot::test_to_dict`
- `TestPoolStatsSnapshot::test_frozen`
- `TestQueryOrderEntry::test_to_dict`
- `TestQueryOrderEntry::test_frozen`
- `TestPoolExecutionContext::test_frozen`
- `TestGateEvaluationContext::test_frozen`
- `TestAggregationFlushContext::test_frozen`

Keep: All `test_canonical_json_*`, `test_from_executor_stats_*`, `test_protocol_conformance_*`, `test_require_int_validation_*` tests.

- [ ] **Step 1: Remove all listed test methods from their classes**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_node_state_context.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_node_state_context.py
git commit -m "test: remove 9 low-value frozen/echo tests from test_node_state_context.py"
```

---

### Task 6: Trim `test_field_contract.py` — Remove Constructor Echo & Frozen Tests

**Files:**
- Modify: `tests/unit/contracts/test_field_contract.py`

Remove:
- `test_create_declared_field`
- `test_create_inferred_field`
- `test_python_type_accepts_primitives`
- `test_frozen_cannot_modify_normalized_name`
- `test_frozen_cannot_modify_original_name`
- `test_frozen_cannot_modify_python_type`
- `test_frozen_cannot_modify_required`
- `test_frozen_cannot_modify_source`
- `test_uses_slots_no_dict`
- `test_cannot_add_arbitrary_attributes`
- `test_source_literal_type_annotation`

Keep: All equality/hashing tests (critical for set/dict cache-key behavior).

- [ ] **Step 1: Remove all listed test functions**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_field_contract.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_field_contract.py
git commit -m "test: remove 11 low-value echo/frozen/slots tests from test_field_contract.py"
```

---

### Task 7: Trim `test_routing.py` — Remove Echo & Frozen Tests

**Files:**
- Modify: `tests/unit/contracts/test_routing.py`

Remove:
- `test_has_mode_field`
- `test_continue_action`
- `test_frozen` (in TestRoutingAction)
- `test_frozen` (in TestRoutingSpec if present)
- `test_frozen` (in TestEdgeInfo if present)
- `test_create_edge_info`
- `test_edge_info_with_copy`
- `test_correct_usage_with_enum`
- `test_create_with_move`
- `test_create_with_copy`

Keep: All `test_*_raises` validation tests, `test_reason_mutation_prevented_by_deep_copy`, `test_reason_deep_copied`, all factory method tests that verify invariants.

- [ ] **Step 1: Remove all listed test functions**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_routing.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_routing.py
git commit -m "test: remove 10 low-value echo/frozen tests from test_routing.py"
```

---

### Task 8: Trim `test_transform_contract.py` — Remove Echo Tests

**Files:**
- Modify: `tests/unit/contracts/test_transform_contract.py`

Remove:
- `test_creates_fixed_contract_from_schema`
- `test_field_types_from_annotations`
- `test_fields_are_declared`
- `test_original_equals_normalized`
- `test_extra_allow_creates_flexible`
- `test_bool_type_preserved`
- `test_bool_field`

Keep: All validation tests (`test_valid_output_returns_empty`, `test_type_mismatch_returns_violation`, etc.), all nullable semantics tests, all regression tests.

- [ ] **Step 1: Remove all listed test functions**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_transform_contract.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_transform_contract.py
git commit -m "test: remove 7 low-value echo tests from test_transform_contract.py"
```

---

### Task 9: Trim `test_errors.py` — Remove TypedDict Echo & Type Introspection Tests

**Files:**
- Modify: `tests/unit/contracts/test_errors.py`

Remove type introspection tests:
- `test_routing_reason_is_union_type`
- `test_routing_reason_variants_are_typed_dicts`
- `test_transform_action_category_values`
- `test_transform_error_category_literal_values`

Remove TypedDict constructor echo tests (pass values, assert same values back — TypedDicts have no runtime validation):
- `test_routing_reason_accepts_config_gate_reason`
- `test_transform_success_reason_has_action_field`
- `test_transform_success_reason_accepts_optional_fields`
- `test_transform_success_reason_accepts_metadata`
- `test_config_gate_reason_construction`
- `test_transform_error_reason_accepts_optional_error_fields`
- `test_minimal_error_reason`
- `test_usage_stats_nested_typeddict`
- `test_template_error_entry_structure`
- `test_row_error_entry_structure`
- `test_usage_stats_partial`
- `test_minimal_query_failure`
- `test_query_failure_with_error`
- `test_minimal_error_detail`
- `test_error_detail_with_context`
- `test_failed_queries_with_strings`
- `test_failed_queries_with_details`
- `test_failed_queries_mixed`
- `test_errors_with_strings`
- `test_errors_with_details`
- `test_errors_mixed`

Remove realistic-but-still-echo pattern tests (also TypedDict constructor echo — just bigger payloads):
- `test_api_error_pattern`
- `test_field_error_pattern`
- `test_llm_truncation_pattern`
- `test_type_mismatch_pattern`
- `test_rate_limit_pattern`
- `test_batch_job_error_pattern`
- `test_batch_template_errors_pattern`
- `test_content_safety_violation_pattern`
- `test_json_parsing_failure_pattern`
- `test_template_rendering_failure_pattern`

Keep: All `__post_init__` validation tests, `to_dict` serialization tests, frozen dataclass behavior (for error types that have custom `__post_init__`), deep freeze tests.

- [ ] **Step 1: Remove all listed test functions**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_errors.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_errors.py
git commit -m "test: remove 31 low-value TypedDict echo/introspection tests from test_errors.py"
```

---

### Task 10: Trim `test_telemetry_config.py` — Remove Enum & Constructor Echo Tests

**Files:**
- Modify: `tests/unit/contracts/test_telemetry_config.py`

Remove enum value checks:
- `test_lifecycle_value`
- `test_rows_value`
- `test_full_value`
- `test_is_string_enum` (for TelemetryGranularity)
- `test_block_value`
- `test_drop_value`
- `test_slow_value`
- `test_is_string_enum` (for BackpressureMode)

Remove constructor echo / frozen tests:
- `test_minimal_config`
- `test_with_options` (ExporterSettings)
- `test_with_options` (TelemetrySettings)
- `test_creation` (ExporterConfig)
- `test_frozen` (ExporterConfig)
- `test_frozen` (TelemetrySettings)
- `test_frozen` (RuntimeTelemetryConfig)
- `test_protocol_fields_accessible`

Keep: All `__post_init__` validation tests, factory method tests (`from_settings_*`), protocol compliance, fail-fast tests.

- [ ] **Step 1: Remove all listed test functions**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_telemetry_config.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_telemetry_config.py
git commit -m "test: remove 16 low-value enum/echo/frozen tests from test_telemetry_config.py"
```

---

### Task 11: Trim `test_new_errors.py` — Remove isinstance & Constructor Echo Tests

**Files:**
- Modify: `tests/unit/contracts/test_new_errors.py`

Remove all instances of these test patterns across the 3 error class test classes:
- `test_construction_and_message` (all 3 instances — one per error class)
- `test_is_exception` (all 3 instances)
- `test_construction_converts_to_tuple`

Keep: All validation tests (`test_empty_*_raises`, `test_none_*_raises`), deep freeze test, mutation isolation test.

- [ ] **Step 1: Remove all listed test functions**
- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_new_errors.py -x -q`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_new_errors.py
git commit -m "test: remove 7 low-value isinstance/echo tests from test_new_errors.py"
```

---

### Task 12: Trim Small Files — Batch Removal

This task covers 1–4 test removals each from multiple small files. Remove the listed functions from each file.

**Files:**
- Modify: `tests/unit/contracts/test_checkpoint.py`
- Modify: `tests/unit/contracts/test_engine_contracts.py`
- Modify: `tests/unit/contracts/test_batch_checkpoint.py`
- Modify: `tests/unit/contracts/test_diversion.py`
- Modify: `tests/unit/contracts/test_diverted_outcome.py`
- Modify: `tests/unit/contracts/test_events.py`
- Modify: `tests/unit/contracts/test_coalesce_enums.py`
- Modify: `tests/unit/contracts/test_coalesce_metadata.py`
- Modify: `tests/unit/contracts/test_probes.py`
- Modify: `tests/unit/contracts/test_results.py`
- Modify: `tests/unit/contracts/test_source_row_contract.py`
- Modify: `tests/unit/contracts/test_contract_violations.py`
- Modify: `tests/unit/contracts/test_config.py`

**test_checkpoint.py** — remove 4:
- `test_resume_point_accepts_typed_aggregation_state`
- `test_resume_point_accepts_none_aggregation_state`
- `test_resume_point_accepts_typed_coalesce_state`
- `test_resume_point_accepts_zero_sequence_number`

**test_engine_contracts.py** — remove 2:
- `test_construction_with_all_fields`
- `test_generic_type_parameter`

**test_batch_checkpoint.py** — remove 2:
- `test_frozen` (in TestRowMappingEntry class)
- `test_frozen_immutability` (in TestBatchCheckpointState class)

**test_diversion.py** — remove 4:
- `test_frozen` (in TestRowDiversion)
- `test_row_data_deep_frozen`
- `test_row_data_nested_frozen`
- `test_frozen` (in TestSinkWriteResult)

**test_diverted_outcome.py** — remove 4:
- `test_diverted_value`
- `test_diverted_is_terminal`
- `test_rows_diverted_default_zero`
- `test_rows_diverted_explicit`

**test_events.py** — remove 4:
- `test_transform_completed_in_contracts`
- `test_gate_evaluated_in_contracts`
- `test_token_completed_in_contracts`
- `test_telemetry_events_inherit_from_contracts_base`

**test_coalesce_enums.py** — remove 1:
- `test_members`

**test_coalesce_metadata.py** — remove 3:
- `test_policy_is_enum`
- `test_merge_strategy_is_enum`
- `test_factory_for_late_arrival_uses_enum`

**test_probes.py** — remove 1:
- `test_compliant_implementation_passes_isinstance`

**test_results.py** — remove 1:
- `test_status_is_literal_not_enum`

**test_source_row_contract.py** — remove 2:
- `test_frozen_rejects_field_reassignment`
- `test_frozen_rejects_quarantine_field_reassignment`

**test_contract_violations.py** — remove 10:
- `test_contract_violation_stores_normalized_name`
- `test_contract_violation_stores_original_name`
- `test_missing_field_stores_names`
- `test_type_mismatch_stores_expected_type`
- `test_type_mismatch_stores_actual_type`
- `test_type_mismatch_stores_actual_value`
- `test_extra_field_stores_names`
- `test_contract_merge_error_stores_field`
- `test_contract_merge_error_stores_type_a`
- `test_contract_merge_error_stores_type_b`

**test_config.py** — remove 1:
- `test_contracts_config_items_exist`

- [ ] **Step 1: Remove all listed test functions from each file**

Work through each file in order. For each file, delete the listed functions. If removing a function empties a test class, delete the class too.

- [ ] **Step 2: Run all contracts tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -x -q`

Expected: All remaining tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_checkpoint.py \
       tests/unit/contracts/test_engine_contracts.py \
       tests/unit/contracts/test_batch_checkpoint.py \
       tests/unit/contracts/test_diversion.py \
       tests/unit/contracts/test_diverted_outcome.py \
       tests/unit/contracts/test_events.py \
       tests/unit/contracts/test_coalesce_enums.py \
       tests/unit/contracts/test_coalesce_metadata.py \
       tests/unit/contracts/test_probes.py \
       tests/unit/contracts/test_results.py \
       tests/unit/contracts/test_source_row_contract.py \
       tests/unit/contracts/test_contract_violations.py \
       tests/unit/contracts/test_config.py
git commit -m "test: remove 39 low-value echo/frozen/enum tests across 13 small contract test files"
```

---

### Task 13: Add `test_run_result.py` — Fill Coverage Gap

**Files:**
- Create: `tests/unit/contracts/test_run_result.py`

`RunResult` (51 LOC) has `__post_init__` validation via `require_int` on all 11 numeric fields, empty `run_id` rejection, and `freeze_fields` on `routed_destinations`. Currently has no dedicated test file.

- [ ] **Step 1: Write the test file**

```python
"""Tests for RunResult — pipeline execution outcome contract.

Validates __post_init__ guards: empty run_id rejection, require_int on all
numeric fields (negative rejection, bool rejection, float rejection),
and freeze_fields on routed_destinations.
"""

import pytest
from types import MappingProxyType

from elspeth.contracts.enums import RunStatus
from elspeth.contracts.run_result import RunResult
from tests.fixtures.factories import make_run_result


class TestRunResultValidation:
    """__post_init__ guards on RunResult."""

    def test_empty_run_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="run_id must not be empty"):
            RunResult(
                run_id="",
                status=RunStatus.COMPLETED,
                rows_processed=0,
                rows_succeeded=0,
                rows_failed=0,
                rows_routed=0,
            )

    @pytest.mark.parametrize(
        "field",
        [
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_buffered",
            "rows_diverted",
        ],
    )
    def test_negative_value_rejected(self, field: str) -> None:
        """Every numeric field must be >= 0."""
        kwargs = {
            "run_id": "run-1",
            "status": RunStatus.COMPLETED,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "rows_routed": 0,
            field: -1,
        }
        with pytest.raises(ValueError, match=field):
            RunResult(**kwargs)

    @pytest.mark.parametrize(
        "field",
        [
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed",
            "rows_quarantined",
            "rows_forked",
            "rows_coalesced",
            "rows_coalesce_failed",
            "rows_expanded",
            "rows_buffered",
            "rows_diverted",
        ],
    )
    def test_bool_rejected(self, field: str) -> None:
        """Bool must not be accepted as int (Python subclass trap)."""
        kwargs = {
            "run_id": "run-1",
            "status": RunStatus.COMPLETED,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "rows_routed": 0,
            field: True,
        }
        with pytest.raises(TypeError):
            RunResult(**kwargs)

    @pytest.mark.parametrize(
        "field",
        [
            "rows_processed",
            "rows_succeeded",
            "rows_failed",
            "rows_routed",
        ],
    )
    def test_float_rejected(self, field: str) -> None:
        """Float must not be silently accepted for int fields."""
        kwargs = {
            "run_id": "run-1",
            "status": RunStatus.COMPLETED,
            "rows_processed": 0,
            "rows_succeeded": 0,
            "rows_failed": 0,
            "rows_routed": 0,
            field: 1.5,
        }
        with pytest.raises(TypeError):
            RunResult(**kwargs)


class TestRunResultImmutability:
    """Frozen dataclass + freeze_fields on routed_destinations."""

    def test_routed_destinations_frozen(self) -> None:
        """Dict passed to routed_destinations must be deep-frozen."""
        result = make_run_result(routed_destinations={"sink_a": 5, "sink_b": 3})
        assert isinstance(result.routed_destinations, MappingProxyType)

    def test_routed_destinations_default_is_empty_frozen(self) -> None:
        """Default routed_destinations must be an empty frozen mapping."""
        result = make_run_result()
        assert isinstance(result.routed_destinations, MappingProxyType)
        assert len(result.routed_destinations) == 0

    def test_routed_destinations_mutation_blocked(self) -> None:
        """Callers must not be able to mutate routed_destinations after creation."""
        result = make_run_result(routed_destinations={"sink_a": 5})
        with pytest.raises(TypeError):
            result.routed_destinations["sink_b"] = 10  # type: ignore[index]


class TestRunResultFactory:
    """Tests for the make_run_result factory (ensures factory is usable)."""

    def test_factory_defaults_produce_valid_result(self) -> None:
        result = make_run_result()
        assert result.status == RunStatus.COMPLETED
        assert result.rows_processed == 10

    def test_factory_accepts_all_overrides(self) -> None:
        result = make_run_result(
            run_id="custom-run",
            status=RunStatus.FAILED,
            rows_processed=100,
            rows_succeeded=90,
            rows_failed=10,
            rows_routed=5,
            rows_quarantined=2,
            rows_forked=3,
            rows_coalesced=1,
            rows_coalesce_failed=1,
            rows_expanded=4,
            rows_buffered=2,
            routed_destinations={"x": 5},
        )
        assert result.run_id == "custom-run"
        assert result.status == RunStatus.FAILED
        assert result.rows_failed == 10
        assert result.routed_destinations["x"] == 5
```

- [ ] **Step 2: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_run_result.py -v`

Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_run_result.py
git commit -m "test: add RunResult validation tests — require_int guards, immutability, factory"
```

---

### Task 14: Add `test_plugin_context_recording.py` — Fill Coverage Gap

**Files:**
- Create: `tests/unit/contracts/test_plugin_context_recording.py`

Tests for `PluginContext.record_validation_error()` and `record_transform_error()` guard clauses. These methods have FrameworkBugError guards for missing landscape and missing node_id, plus the happy path delegates to landscape.

- [ ] **Step 1: Write the test file**

```python
"""Tests for PluginContext.record_validation_error() and record_transform_error().

Tests the offensive programming guards (FrameworkBugError) and basic delegation
to LandscapeRecorder. Uses make_source_context() for real landscape integration
and manual PluginContext construction for guard-clause tests.
"""

from unittest.mock import Mock

import pytest

from elspeth.contracts import FrameworkBugError
from elspeth.contracts.plugin_context import (
    PluginContext,
    TransformErrorToken,
    ValidationErrorToken,
)
from tests.fixtures.factories import make_source_context


class TestRecordValidationErrorGuards:
    """record_validation_error() must crash on missing landscape or node_id."""

    def test_raises_when_landscape_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, node_id="source")
        with pytest.raises(FrameworkBugError, match="record_validation_error.*without landscape"):
            ctx.record_validation_error(
                row={"name": "test"},
                error="field X is NULL",
                schema_mode="fixed",
                destination="discard",
            )

    def test_raises_when_node_id_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=Mock(), node_id=None)
        with pytest.raises(FrameworkBugError, match="record_validation_error.*without node_id"):
            ctx.record_validation_error(
                row={"name": "test"},
                error="field X is NULL",
                schema_mode="fixed",
                destination="discard",
            )


class TestRecordValidationErrorHappyPath:
    """record_validation_error() delegates to landscape and returns token."""

    def test_returns_validation_error_token(self) -> None:
        """Happy path: row with id field → token with that row_id."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row={"id": "row-42", "name": "test"},
            error="field X is NULL",
            schema_mode="fixed",
            destination="discard",
        )
        assert isinstance(token, ValidationErrorToken)
        assert token.row_id == "row-42"
        assert token.node_id == "source"
        assert token.destination == "discard"
        assert token.error_id is not None  # Landscape assigns an error_id

    def test_row_without_id_uses_content_hash(self) -> None:
        """Row without 'id' field → row_id derived from stable_hash."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row={"name": "test"},
            error="missing required field",
            schema_mode="flexible",
            destination="quarantine_sink",
        )
        assert isinstance(token, ValidationErrorToken)
        assert len(token.row_id) == 16  # stable_hash[:16]
        assert token.destination == "quarantine_sink"

    def test_non_dict_row_uses_repr_hash(self) -> None:
        """Non-dict row (e.g., JSON primitive) → row_id from repr_hash."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row="not a dict",
            error="expected dict, got str",
            schema_mode="parse",
            destination="discard",
        )
        assert isinstance(token, ValidationErrorToken)
        assert len(token.row_id) == 16

    def test_custom_destination_propagated(self) -> None:
        """Destination string flows through to the returned token."""
        ctx = make_source_context()
        token = ctx.record_validation_error(
            row={"id": "row-1"},
            error="bad data",
            schema_mode="fixed",
            destination="error_sink",
        )
        assert token.destination == "error_sink"


class TestRecordTransformErrorGuards:
    """record_transform_error() must crash on missing landscape."""

    def test_raises_when_landscape_is_none(self) -> None:
        ctx = PluginContext(run_id="run-1", config={}, landscape=None, node_id="transform-1")
        with pytest.raises(FrameworkBugError, match="record_transform_error.*without landscape"):
            ctx.record_transform_error(
                token_id="tok-1",
                transform_id="transform-1",
                row={"data": "test"},
                error_details={"action": "quarantine", "reason": "API returned 500"},
                destination="discard",
            )


class TestRecordTransformErrorHappyPath:
    """record_transform_error() delegates to landscape and returns token."""

    def test_returns_transform_error_token(self) -> None:
        """Happy path: recording succeeds → returns TransformErrorToken."""
        ctx = make_source_context()
        # record_transform_error doesn't need operation_id — it uses landscape directly
        token = ctx.record_transform_error(
            token_id="tok-1",
            transform_id="transform-1",
            row={"data": "test"},
            error_details={"action": "quarantine", "reason": "API returned 500"},
            destination="error_sink",
        )
        assert isinstance(token, TransformErrorToken)
        assert token.token_id == "tok-1"
        assert token.transform_id == "transform-1"
        assert token.destination == "error_sink"
        assert token.error_id is not None
```

- [ ] **Step 2: Run the new tests**

Run: `.venv/bin/python -m pytest tests/unit/contracts/test_plugin_context_recording.py -v`

Expected: All pass. If `record_transform_error` fails because `LandscapeRecorder.record_transform_error()` doesn't exist yet or has a different signature, adapt the test to match the actual recorder API — check `src/elspeth/core/landscape/recorder.py` for the method signature.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/contracts/test_plugin_context_recording.py
git commit -m "test: add PluginContext recording guard + happy-path tests"
```

---

### Task 15: Final Verification & Count

- [ ] **Step 1: Run full contracts test suite**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ -x -q`

Expected: All pass, 0 failures.

- [ ] **Step 2: Count tests and compare to baseline**

Run: `.venv/bin/python -m pytest tests/unit/contracts/ --co -q 2>&1 | tail -3`

Expected: Roughly 2204 - ~155 removed + ~30 added ≈ ~2079 tests. The exact number will depend on parametrized test expansion. Record the final count.

- [ ] **Step 3: Run full project tests to check for cross-test dependencies**

Run: `.venv/bin/python -m pytest tests/ -x -q --timeout=120`

Expected: All pass. No other test should depend on the removed low-value tests.

- [ ] **Step 4: Commit any fixups if needed**

Only if Step 3 reveals issues.
