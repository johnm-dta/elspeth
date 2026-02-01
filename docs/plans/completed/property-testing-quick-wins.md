# Property Testing Quick Wins Plan

**Goal:** Rapidly expand property test coverage to high-risk areas using existing infrastructure.

**Status:** ✅ **COMPLETED** (2026-01-29)

---

## Executive Summary

| Day | Focus Area | Planned | Actual | Status |
|-----|-----------|---------|--------|--------|
| **Day 1** | Field Normalization | 4 tests | 10 tests | ✅ Exceeded |
| **Day 1** | Payload Store | 3 tests | 9 tests | ✅ Exceeded |
| **Day 2** | Token Fork Isolation | 2 tests | 6 tests | ✅ Exceeded |
| **Day 2** | Retry Config | 2 tests | 11 tests | ✅ Exceeded |
| **Day 3** | Integration & Polish | 2 tests | 7 tests | ✅ Exceeded |

**Planned:** 13 new property tests
**Actual:** 43 new property tests (231% of target)

**Total property tests in suite:** 226 (up from 183 existing)

---

## Implementation Results

### Files Created

| File | Tests | Lines | Status |
|------|-------|-------|--------|
| `tests/property/conftest.py` | - | 112 | ✅ Created |
| `tests/property/sources/__init__.py` | - | 2 | ✅ Created |
| `tests/property/sources/test_field_normalization_properties.py` | 10 | 175 | ✅ Created |
| `tests/property/core/__init__.py` | - | 2 | ✅ Created |
| `tests/property/core/test_payload_store_properties.py` | 9 | 170 | ✅ Created |
| `tests/property/engine/__init__.py` | - | 2 | ✅ Created |
| `tests/property/engine/test_token_properties.py` | 6 | 220 | ✅ Created |
| `tests/property/engine/test_retry_properties.py` | 11 | 165 | ✅ Created |
| `tests/property/integration/__init__.py` | - | 2 | ✅ Created |
| `tests/property/integration/test_cross_module_properties.py` | 7 | 145 | ✅ Created |

### Test Breakdown by Module

#### Field Normalization (10 tests)
- `test_normalize_field_name_is_idempotent` - f(f(x)) == f(x)
- `test_normalize_produces_valid_identifier` - output.isidentifier()
- `test_normalize_never_produces_keywords` - no bare keywords
- `test_normalize_handles_all_keywords` - keyword → keyword_
- `test_collision_detection_is_order_independent` - symmetry
- `test_unique_headers_never_collide` - no false positives
- `test_resolve_preserves_header_count` - len(in) == len(out)
- `test_resolve_mapping_covers_all_inputs` - complete mapping
- `test_resolve_columns_mode_passthrough` - columns unchanged
- `test_resolve_without_normalization_passthrough` - passthrough mode

#### Payload Store (9 tests)
- `test_store_retrieve_roundtrip` - retrieve(store(x)) == x
- `test_exists_after_store` - exists() returns True
- `test_store_hash_is_deterministic` - same content = same hash
- `test_store_hash_matches_sha256` - standard SHA-256
- `test_different_content_different_hash` - collision resistance
- `test_store_is_idempotent` - no duplicate files
- `test_store_then_delete_then_store` - delete/re-store works
- `test_retrieve_nonexistent_raises_keyerror` - proper error
- `test_corrupted_content_detected` - IntegrityError on corruption

#### Token Fork Isolation (6 tests)
- `test_fork_creates_isolated_copies` - mutation isolation
- `test_fork_isolates_deeply_nested_data` - deep structure isolation
- `test_fork_preserves_parent_data` - parent unchanged
- `test_fork_children_have_correct_metadata` - correct token attrs
- `test_fork_with_override_uses_override` - explicit data used
- `test_fork_without_override_uses_parent_data` - default behavior

#### Retry Config (11 tests)
- `test_valid_config_construction` - valid configs accepted
- `test_invalid_max_attempts_rejected` - max_attempts < 1 rejected
- `test_config_fields_are_immutable_readable` - field access works
- `test_no_retry_is_single_attempt` - no_retry() → max_attempts=1
- `test_from_policy_always_produces_valid_config` - coercion safety
- `test_from_policy_none_returns_no_retry` - None → no_retry
- `test_from_policy_empty_dict_uses_defaults` - defaults applied
- `test_from_policy_preserves_valid_values` - valid values preserved
- `test_negative_max_attempts_coerced_to_minimum` - coercion
- `test_negative_base_delay_coerced_to_minimum` - coercion
- `test_negative_jitter_coerced_to_zero` - coercion

#### Cross-Module Integration (7 tests)
- `test_normalized_fields_hash_consistently` - normalization + hash
- `test_normalization_canonical_idempotence` - normalize chain
- `test_canonical_payload_hash_consistency` - canonical + store
- `test_canonical_stable_hash_matches_payload_store` - stable_hash
- `test_payload_store_preserves_canonical_json_exactly` - byte exact
- `test_full_pipeline_determinism` - end-to-end determinism
- `test_hash_proves_integrity` - integrity verification

---

## Shared Strategies Created

The `tests/property/conftest.py` file provides reusable Hypothesis strategies:

| Strategy | Purpose | Used By |
|----------|---------|---------|
| `json_primitives` | RFC 8785 safe primitives | row_data, integration |
| `json_values` | Recursive nested JSON | integration |
| `row_data` | Transform-like data | all modules |
| `messy_headers` | External header simulation | field normalization |
| `normalizable_headers` | Headers that normalize | field normalization |
| `mutable_nested_data` | For isolation tests | token fork |
| `deeply_nested_data` | Stress test deepcopy | token fork |
| `binary_content` | Arbitrary bytes | payload store |
| `nonempty_binary` | Non-empty bytes | payload store |
| `small_binary` | Fast test bytes | payload store |
| `valid_max_attempts` | Retry config values | retry |
| `valid_delays` | Delay floats | retry |
| `valid_jitter` | Jitter floats | retry |
| `branch_names` | Fork branch names | token |
| `unique_branches` | Unique branch lists | token |
| `multiple_branches` | 2+ branches | token |

---

## Issues Discovered During Implementation

### 1. Keyword Handling Edge Case (Test Bug)

**Discovery:** Property test found that `False` → `false` (not a keyword after lowercase)

**Root Cause:** Test assumed keywords normalize to `keyword_`, but normalization lowercases first. Python's `False` lowercases to `false`, which is NOT a Python keyword.

**Resolution:** Fixed test to check if the *lowercased* form is still a keyword.

**Impact:** Test design bug, not code bug. Existing code is correct.

### 2. RFC 8785 Safe Integer Boundary (Test Bug)

**Discovery:** `9007199254740992.0` caused IntegerDomainError after JSON round-trip

**Root Cause:** Test used `json.loads()` which converts edge floats to ints, then tried to re-canonicalize the int (outside safe domain).

**Resolution:** Changed test to verify byte-identical storage (what audit integrity actually requires) rather than Python object round-trip.

**Impact:** Test design bug, not code bug. RFC 8785 boundary handling is correct.

---

## Verification Checklist

- [x] All tests pass with `pytest tests/property/ -v`
- [x] CI profile runs 100 examples (default)
- [x] Nightly profile available (1000 examples)
- [x] No flaky tests (verified with multiple runs)
- [x] Strategies are reusable in `conftest.py`
- [x] Each test has clear property description
- [x] Edge cases discovered are documented

**Final verification run:**
```
$ pytest tests/property/ -v
226 passed in 35.25s
```

---

## Risk Reduction Achieved

| Risk Area | Property Coverage | ELSPETH Trust Tier |
|-----------|-------------------|-------------------|
| Trust boundary coercion | Idempotence, validity | Tier 3 (external data) |
| Content-addressable storage | Round-trip, determinism | Tier 1 (audit trail) |
| Audit trail data isolation | Fork isolation, parent preservation | Tier 1 (audit trail) |
| Configuration validation | Construction, coercion | Tier 3 (external config) |
| Cross-module consistency | Hash pipeline integrity | Tier 1 (audit trail) |

---

## Future Expansion Opportunities

Areas identified but not yet implemented:

1. **Checkpoint Recovery** - Save/restore round-trip properties
2. **Expression Parser** - Parse → AST → evaluate consistency
3. **DAG Validation** - Acyclicity, topological sort invariants
4. **Coalesce Operations** - Merge policy properties
5. **Rate Limiting** - Token bucket invariants

---

## Conclusion

The quick wins plan was completed successfully with 231% of the planned test coverage. The property tests now provide systematic verification of:

1. **Field Normalization** - Trust boundary handling at source ingestion
2. **Payload Store** - Content-addressable storage integrity
3. **Token Fork** - Audit trail data isolation (deepcopy verification)
4. **Retry Config** - Configuration validation and coercion
5. **Cross-Module** - Hash consistency across the audit pipeline

The two edge cases discovered during implementation were both test design issues, validating that the existing production code handles these cases correctly. This is exactly the value property testing provides - it found assumptions in the test design that didn't match the actual (correct) behavior.
