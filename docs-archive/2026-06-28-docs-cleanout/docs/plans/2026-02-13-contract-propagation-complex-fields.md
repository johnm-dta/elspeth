# Contract Propagation Complex-Field Preservation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve complex transform output fields (`dict`/`list`) in propagated contracts as `python_type=object` instead of silently dropping them.

**Architecture:** Keep the change tightly scoped to contract propagation logic, where type inference currently skips unsupported complex values. We will preserve existing behavior for invalid non-finite numeric values (NaN/Infinity), and change only `dict`/`list` `TypeError` paths from “skip” to “infer as object.” We will lock this with targeted unit tests in both `propagate_contract()` and `narrow_contract_to_output()`, plus one integration test that exercises real source->transform contract propagation.

**Tech Stack:** Python 3.13, pytest, dataclass-based schema contracts.

**Prerequisites:**
- `.venv` exists and can run pytest/ruff.
- Working branch is clean before edits.
- No schema migration required.

---

### Task 1: Add Failing Tests That Capture the Bug

**Files:**
- Modify: `tests/unit/contracts/test_contract_propagation.py`
- Modify: `tests/unit/contracts/test_contract_narrowing.py`

**Step 1: Update non-primitive contract propagation expectations**

Add/adjust tests in `TestPropagateContractNonPrimitiveTypes` so dict/list fields are expected in output contracts with `python_type is object`.

```python
usage_field = next(f for f in output_contract.fields if f.normalized_name == "response_usage")
assert usage_field.python_type is object
```

```python
tags_field = next(f for f in output_contract.fields if f.normalized_name == "tags")
assert tags_field.python_type is object
```

**Why this test:** Current tests intentionally encode the buggy skip behavior; we need to flip them to intended behavior.

**Step 2: Update narrowing tests for non-primitive additions**

Change `test_narrow_contract_skips_non_primitive_types` so dict/list fields are expected to be present as inferred `object` fields.

```python
assert {f.normalized_name for f in result.fields} == {"a", "dict_field", "list_field"}
assert next(f for f in result.fields if f.normalized_name == "dict_field").python_type is object
assert next(f for f in result.fields if f.normalized_name == "list_field").python_type is object
```

**Why this test:** `narrow_contract_to_output()` has the same skip bug and must match propagation semantics.

**Step 3: Add boundary tests for non-finite and non-dict/list unsupported types**

Add tests that lock unchanged behavior:
- `propagate_contract()` still skips unsupported non-dict/list types (e.g., custom object)
- `propagate_contract()` still raises on non-finite floats (`ValueError`)
- `narrow_contract_to_output()` still skips unsupported non-dict/list types and non-finite floats

```python
class _CustomUnsupported: ...
output_row = {"id": 1, "custom": _CustomUnsupported()}
```

```python
output_row = {"id": 1, "bad": float("nan")}
with pytest.raises(ValueError):
    propagate_contract(...)
```

**Why these tests:** They constrain scope to dict/list only and prevent unintended semantic broadening.

**Step 4: Run tests to verify RED state**

Run:
` .venv/bin/python -m pytest tests/unit/contracts/test_contract_propagation.py tests/unit/contracts/test_contract_narrowing.py -q`

Expected output:
- Failing assertions showing non-primitive fields are currently skipped.

**Definition of Done:**
- [ ] Tests explicitly assert `object` typing for dict/list additions
- [ ] Tests explicitly preserve non-dict/list and NaN/Infinity behavior
- [ ] Tests fail pre-fix for the right reason

---

### Task 2: Implement TypeError -> object Fallback in Contract Propagation

**Files:**
- Modify: `src/elspeth/contracts/contract_propagation.py`

**Step 1: Update `propagate_contract()` inference path**

Current behavior:
- `TypeError` => `continue` (field omitted)

Planned behavior:
- `TypeError` + `value is dict/list` => `python_type = object`
- `TypeError` + other value types => keep existing skip behavior

```python
try:
    python_type = normalize_type_for_contract(value)
except TypeError:
    if isinstance(value, (dict, list)):
        python_type = object
    else:
        continue
```

**Step 2: Update `narrow_contract_to_output()` inference path**

Current behavior:
- `(TypeError, ValueError)` => skip/log

Planned behavior:
- `TypeError` + `value is dict/list` => `python_type = object`
- `TypeError` + other value types => keep existing skip/log behavior
- `ValueError` => keep existing skip/log behavior (preserves current NaN/Infinity handling)

```python
try:
    python_type = normalize_type_for_contract(value)
except TypeError:
    if isinstance(value, (dict, list)):
        python_type = object
    else:
        skipped_fields.append(name)
        log.debug(...)
        continue
except ValueError as e:
    skipped_fields.append(name)
    log.debug(...)
    continue
```

**Step 3: Keep metadata behavior unchanged**

- `required=False` for inferred new fields
- `source="inferred"`
- No changes to rename-preservation logic

**Why this implementation:** Lowest-risk surgical fix; preserves all existing behavior except the bug-causing skip on complex types.

**Step 4: Update stale comments that describe old skip behavior**

- Update comment in `src/elspeth/plugins/llm/base.py` that currently says complex fields are intentionally skipped.

**Step 5: Run focused tests to verify GREEN state**

Run:
` .venv/bin/python -m pytest tests/unit/contracts/test_contract_propagation.py tests/unit/contracts/test_contract_narrowing.py -q`

Expected output:
- All targeted tests pass.

**Definition of Done:**
- [ ] Complex fields are represented in contract with `python_type=object`
- [ ] Targeted contract tests pass

---

### Task 3: Regression and Compatibility Checks

**Files:**
- No additional source changes expected

**Step 1: Validate LLM contract compatibility**

Run:
` .venv/bin/python -m pytest tests/unit/plugins/llm/test_llm_transform_contract.py -k usage -q`

**Why:** LLM currently has explicit `_usage` fallback logic; ensure this remains correct and non-duplicative.

**Step 2: Run broader contract suite smoke tests**

Run:
` .venv/bin/python -m pytest tests/unit/contracts -q`

**Why:** Detect side effects in schema contract workflows.

**Step 3: Lint touched files**

Run:
` .venv/bin/python -m ruff check src/elspeth/contracts/contract_propagation.py src/elspeth/plugins/llm/base.py tests/unit/contracts/test_contract_propagation.py tests/unit/contracts/test_contract_narrowing.py`

**Step 4: Add one integration test for real contract propagation path**

**Files:**
- Modify: `tests/integration/plugins/transforms/test_contract.py`

Add test verifying `propagate_contract()` preserves a newly added dict field as `object` from a real source row contract.

Run:
` .venv/bin/python -m pytest tests/integration/plugins/transforms/test_contract.py -k added_dict -q`

**Definition of Done:**
- [ ] LLM usage-related tests pass
- [ ] Unit contract suite passes
- [ ] Integration transform contract test for dict field passes
- [ ] Ruff checks pass

---

## Risk/Complexity Gate

### Complexity Assessment
- Code complexity: Low (two local inference branches)
- Test complexity: Low-to-medium (expectation updates + one behavioral expansion)
- Integration complexity: Low (no API/schema migration)

### Risk Assessment
- Functional risk: Medium-low (changes contract contents for complex fields)
- Data integrity risk: Low-positive (contracts better reflect real outputs)
- Performance risk: Low (constant-time type assignment)
- Reversibility: High (single module rollback)

### Go/No-Go Criteria
- **GO** if targeted and regression suites pass, with only intended contract changes.
- **NO-GO** if broad contract tests reveal downstream assumptions that require architectural redesign rather than localized fix.
