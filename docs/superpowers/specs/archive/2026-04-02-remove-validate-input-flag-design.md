# Remove `validate_input` opt-in flag — unconditional executor validation

**Date:** 2026-04-02
**Status:** Draft
**Scope:** Plugin protocols, base classes, executors, all plugins, tests

## Problem

The executor validates transform/sink input against `input_schema` only when the
plugin sets `validate_input = True`. This flag defaults to `False` in both base
classes (`BaseTransform`, `BaseSink`) and is opt-in per-plugin via config.

This directly contradicts the Tier 2 trust model: post-source data has validated
types, and wrong types at a transform or sink boundary are upstream plugin bugs
that must crash immediately. A transform like FieldMapper that only copies/renames
fields performs no value-level operation that would naturally fail on bad types,
so schema violations pass through silently unless the flag is explicitly enabled.

The same gap exists in every plugin that inherits the `False` default: passthrough,
truncate, keyword_filter, batch_stats, json_explode, field_collision, all web_scrape
transforms, csv_sink, json_sink, azure_blob_sink, chroma_sink, and dataverse sink.

Only two plugins (database_sink and the just-fixed field_mapper) hardcode `True`.

## Decision

Remove `validate_input` entirely. The executor validates unconditionally.

**Rationale:**
- Tier 2 contract says wrong types are bugs — there is no valid reason to skip validation
- The flag was a backwards-compatibility escape hatch; per No Legacy Code Policy, we have no users and no compat obligations
- An opt-in safety mechanism is the wrong default for an auditability framework — safety must be opt-out at most, and here there is no valid opt-out case
- Dynamic schemas (`mode: observed`) accept everything, so unconditional validation is a no-op for them — no special-casing needed

## Production code changes

### 1. Protocols (`contracts/plugin_protocols.py`)

Remove `validate_input: bool` from both `TransformProtocol` (line 222) and
`SinkProtocol` (line 448), along with the associated comments.

### 2. Base classes (`plugins/infrastructure/base.py`)

Remove `validate_input: bool = False` from both `BaseTransform` (line 163) and
`BaseSink` (line 415), along with the associated comments.

### 3. Transform executor (`engine/executors/transform.py`)

Remove the `if transform.validate_input:` guard at line 228. The validation
block becomes unconditional:

```python
# --- INPUT VALIDATION (pre-execution) ---
# Validate input against input_schema before calling process().
# Wrong types at a transform boundary are upstream plugin bugs (Tier 2).
try:
    transform.input_schema.model_validate(input_dict)
except ValidationError as e:
    raise PluginContractViolation(
        f"Transform '{transform.name}' input validation failed: {e}. "
        f"This indicates an upstream transform/source schema bug."
    ) from e
```

The `from pydantic import ValidationError` was previously deferred inside the
`if` guard. With the guard removed, hoist it to the module-level imports.

### 4. Sink executor (`engine/executors/sink.py`)

Same pattern — remove the `if sink.validate_input:` guard at line 294. Validation
becomes unconditional for every row batch before `sink.write()`.

### 5. Plugins with explicit `validate_input`

| Plugin | File | Change |
|--------|------|--------|
| field_mapper | `plugins/transforms/field_mapper.py` | Remove `self.validate_input = True` from `__init__`, update docstrings |
| passthrough | `plugins/transforms/passthrough.py` | Remove `validate_input` from config class and `__init__`, update docstrings |
| database_sink | `plugins/sinks/database_sink.py` | Remove `self.validate_input = True` from `__init__` |
| csv_sink | `plugins/sinks/csv_sink.py` | Remove `validate_input` from config class and `__init__`, update docstrings |
| json_sink | `plugins/sinks/json_sink.py` | Remove `validate_input` from config class and `__init__`, update docstrings |

### 6. Plugins with no explicit mention (inherit from base)

No code changes needed. These transforms and sinks previously inherited
`validate_input = False` from the base class. After the base class attribute
is removed, they simply get validated unconditionally by the executor:

- Transforms: truncate, web_scrape, web_scrape_fingerprint, web_scrape_errors,
  web_scrape_extraction, batch_replicate, keyword_filter, batch_stats,
  json_explode, field_collision
- Sinks: azure_blob_sink, chroma_sink, dataverse

### 7. Not changed

`config.py` has three Pydantic `@field_validator` methods named `validate_input`
(lines 446, 519, 888). These are unrelated — they are validator decorators for
config fields, not the plugin validation flag. No changes needed.

## Test changes

### Tests to delete (assert the bug as a feature)

| File | Test | Reason |
|------|------|--------|
| `test_executors.py` | `test_validate_input_disabled_passes_wrong_type` | Tests that wrong types pass through — that's the bug |
| `test_passthrough.py` | `test_validate_input_disabled_passes_wrong_type` | Same |
| `test_passthrough.py` | `test_validate_input_attribute_set_from_config` | Flag no longer exists |
| `test_passthrough.py` | `test_validate_input_skipped_for_dynamic_schema` | Rewrite: validation runs but dynamic schema accepts all |

### Tests to update

| File | Change |
|------|--------|
| `test_executors.py` | Remove `validate_input = False` assignments on mock transforms/sinks; remove `validate_input` from mock specs; update `test_validate_input_rejects_wrong_type` to not set the flag |
| `test_executors.py` | `test_sink_validate_input_rejects_wrong_type` — remove flag set, validation is now unconditional |
| `test_sink_executor_diversion.py` | Remove `sink.validate_input = False` assignments |
| `test_sink_executor_diversion_properties.py` | Remove `sink.validate_input = False` assignments |
| `test_durability.py` | Remove `sink.validate_input = False` |
| `test_csv_sink_properties.py` | Delete `test_csv_sink_validate_input_attribute_set_from_config` |
| `test_json_sink_properties.py` | Delete `test_json_sink_validate_input_attribute_set_from_config` |
| `test_csv_sink_contract.py` | Delete `test_strict_schema_sets_validate_input_for_executor` |
| `test_passthrough_contract.py` | Remove `validate_input` from config; update or delete `test_strict_passthrough_sets_validate_input_for_executor` |
| `test_protocols.py` | Remove `validate_input: bool = False` from mock protocol implementations |
| `test_field_mapper.py` | Already updated — `test_validate_input_always_enabled` asserts `True`; after removal, replace with test that validation is unconditional (no attribute check needed) |
| `tests/fixtures/base_classes.py` | Remove `validate_input: bool = False` from test base class |

### New test

Add `test_unconditional_input_validation` to the executor test suite. Construct
a transform with a fixed schema, feed it a row with wrong types, and assert
`PluginContractViolation` — without ever setting any flag. This is the canonical
"Tier 2 contract enforced" test.

## Performance consideration

Dynamic schemas (`extra="allow"`, no required fields) pass `model_validate()`
with near-zero cost per row. No special-casing or short-circuit needed. If
profiling later shows a hot path, the optimization would be in Pydantic's
model generation, not in re-adding the flag.

## Risk

**Low.** The change makes validation stricter, not looser. Any pipeline that
breaks was already silently forwarding bad types — the failure is a bug surfaced,
not a regression introduced. No external API or config format changes (the
`validate_input` config key will be rejected by Pydantic's `extra="forbid"`
on affected plugin configs, producing a clear error message).
