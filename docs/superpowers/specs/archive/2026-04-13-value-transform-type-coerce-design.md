# Value Transform and Type Coerce Plugins Design

**Date:** 2026-04-13
**Status:** Approved
**Author:** Claude Code + John Morrissey

## Overview

Two new transform plugins for deterministic row-level value manipulation:

| Plugin | Purpose | Coercion | Expressions |
|--------|---------|----------|-------------|
| `value_transform` | Compute/modify field values | No | Yes (ExpressionParser) |
| `type_coerce` | Normalize field types | Yes (strict, explicit) | No |

These plugins provide deterministic alternatives to LLM-based transforms for simple data manipulation tasks like string concatenation, arithmetic, and type normalization.

## Design Principles

1. **Reuse existing infrastructure** — `value_transform` uses the existing `ExpressionParser` (no modifications needed)
2. **Explicit coercion boundary** — Type conversions are a separate, auditable step
3. **Atomic row semantics** — Any operation failure routes the entire original row to error
4. **No fabrication** — Missing fields are errors, not defaults
5. **Strict by default** — No silent truncation, no Python truthiness surprises

## Terminology

- **`schema`**: The YAML config key for schema configuration (matches existing plugins)
- **Schema modes**: `observed` (infer from data), `fixed` (exact fields), `flexible` (at least these fields)

## Plugin 1: `value_transform`

### Purpose

Apply expressions to compute new or modified field values using the existing secure `ExpressionParser`.

### Configuration

```yaml
- plugin: value_transform
  options:
    schema:
      mode: observed
    operations:
      - target: total
        expression: "row['price'] * row['quantity']"
      - target: line
        expression: "row['line'] + ' World'"
      - target: discount_price
        expression: "row['price'] * 0.9"
```

### Configuration Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema` | object | Yes | Schema configuration (mode: observed/fixed/flexible) |
| `operations` | list | Yes | List of operations to apply |
| `operations[].target` | string | Yes | Output field name (valid identifier) |
| `operations[].expression` | string | Yes | Expression to evaluate via ExpressionParser |

### Execution Model

1. Create a **working copy** of the input row
2. For each operation in order:
   - Evaluate `expression` against working copy via `ExpressionParser`
   - Write result to `target` in working copy (create or overwrite)
3. If **all operations succeed**: emit working copy as success
4. If **any operation fails**: emit **original input row** as error (no partial mutations)

### Field Behavior

| Scenario | Behavior |
|----------|----------|
| Target doesn't exist | Created |
| Target exists | Overwritten |
| Self-reference (`row['x']` → `x`) | Valid, overwrites |
| Duplicate targets in list | Sequential rewrites, both execute |
| Missing source field (`row['missing']`) | `ExpressionEvaluationError` → row errors |

### Expression Capabilities

The `ExpressionParser` supports (no modifications needed):

- **Subscript access**: `row['field']`, `row['a']['b']`
- **Method**: `row.get('field')` (single-arg only — defaults forbidden)
- **Safe builtins**: `len()`, `abs()`
- **Arithmetic**: `+`, `-`, `*`, `/`, `//`, `%`
- **Comparisons**: `==`, `!=`, `<`, `>`, `<=`, `>=`
- **Boolean operators**: `and`, `or`, `not`
- **Membership**: `in`, `not in`
- **Identity**: `is`, `is not` (for None checks)
- **Literals**: strings, numbers, booleans, None, list/dict/tuple/set
- **Ternary**: `x if condition else y`

**Forbidden** (enforced by ExpressionParser):
- Type coercion builtins (`str()`, `int()`, `float()`, `bool()`)
- Lambda expressions
- Comprehensions
- Assignment expressions
- Arbitrary function calls
- Attribute access (except `.get`)

**Note on `row.get()`:** If `row.get('field')` returns `None` and the surrounding expression cannot operate on `None` (e.g., `row.get('discount') * 2`), evaluation fails and the row errors. Use ternary expressions for explicit null handling: `row.get('discount') * 2 if row.get('discount') is not None else 0`.

### Validation (Config Time)

- Each operation requires `target` (valid identifier) and `expression` (non-empty)
- All expressions parsed via `ExpressionParser` at construction
- Syntax/security errors fail config validation immediately
- Duplicate targets allowed (no warning — sequential rewrites are legitimate)
- Field existence validated at runtime only; config validation checks syntax and permitted constructs

### Contract Propagation

- `declared_output_fields` = deduplicated set of all `target` values
- Output contract derived from input contract plus declared output fields
- Existing target fields retain their names; new target fields are added to output contract
- If result types cannot be determined statically, output field types remain `any` until runtime

### Audit Trail

```json
{
  "action": "transformed",
  "fields_modified": ["line"],
  "fields_added": ["total", "discount_price"],
  "operations_applied": 3
}
```

### Example

**Input:**
```json
{"price": 10, "quantity": 2, "line": "Hello"}
```

**Config:**
```yaml
operations:
  - target: total
    expression: "row['price'] * row['quantity']"
  - target: line
    expression: "row['line'] + ' World'"
  - target: discount_price
    expression: "row['price'] * 0.9"
```

**Output:**
```json
{"price": 10, "quantity": 2, "line": "Hello World", "total": 20, "discount_price": 9.0}
```

## Plugin 2: `type_coerce`

### Purpose

Perform explicit, strict, per-field type normalization on existing row fields.

### Configuration

```yaml
- plugin: type_coerce
  options:
    schema:
      mode: observed
    conversions:
      - field: price
        to: float
      - field: quantity
        to: int
      - field: is_active
        to: bool
      - field: user_id
        to: str
```

### Configuration Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema` | object | Yes | Schema configuration (mode: observed/fixed/flexible) |
| `conversions` | list | Yes | List of type conversions to apply |
| `conversions[].field` | string | Yes | Field name to convert |
| `conversions[].to` | string | Yes | Target type: `int`, `float`, `bool`, `str` |

### Conversion Rules (Strict)

#### `to: int`

| Input | Result |
|-------|--------|
| `int` | Unchanged |
| `float` with no fractional part (`3.0`) | `3` |
| `float` with fractional part (`3.9`) | **Error** (no silent truncation) |
| String of integer after trim (`"42"`, `" -7 "`) | Parsed int |
| String with decimal (`"3.5"`) | **Error** |
| Scientific notation (`"1e3"`) | **Error** |
| Signed strings (`"+42"`, `"-42"`) | Allowed |
| Empty/whitespace string | **Error** |
| `bool` | **Error** |
| `None` | **Error** |

**Implementation note:** Use `type(value) is int` not `isinstance(value, int)` to prevent `True`/`False` matching as ints.

#### `to: float`

| Input | Result |
|-------|--------|
| `float` | Unchanged (must be finite) |
| `int` | Converted to float |
| Numeric string after trim (`"12.5"`, `" -3.14 "`) | Parsed float |
| Scientific notation (`"1e3"`, `"2.5e-4"`) | Allowed |
| Signed strings (`"+3.14"`, `"-2.5"`) | Allowed |
| `"nan"`, `"inf"`, `"-inf"` | **Error** (non-finite values rejected) |
| Empty/whitespace string | **Error** |
| `bool` | **Error** |
| `None` | **Error** |

**Safety note:** Parsed float values must be finite; `NaN`, `inf`, and `-inf` are rejected to prevent downstream issues in JSON, databases, and aggregations.

#### `to: bool`

| Input | Result |
|-------|--------|
| `bool` | Unchanged |
| `int` `0` | `False` |
| `int` `1` | `True` |
| Other integers (`2`, `-1`) | **Error** |
| String (case-insensitive, trimmed) in **true set**: `true`, `1`, `yes`, `y`, `on` | `True` |
| String (case-insensitive, trimmed) in **false set**: `false`, `0`, `no`, `n`, `off`, `""` | `False` |
| Other strings | **Error** |
| `float` | **Error** |
| `None` | **Error** |

**Rationale:** Python truthiness (`bool("false") == True`) creates silent data corruption. Explicit mapping is safer.

**Scope:** Boolean string mapping is English-only and fixed in V1. Localized variants (e.g., `oui`/`non`) are not supported.

#### `to: str`

| Input | Result |
|-------|--------|
| `str` | Unchanged |
| `int` | Python `str()` (e.g., `42` → `"42"`) |
| `float` | Python `str()` (e.g., `3.14` → `"3.14"`) |
| `bool` | Python `str()` (e.g., `True` → `"True"`) |
| `list`, `dict`, objects, bytes | **Error** (not scalars) |
| `None` | **Error** |

**Scalar definition:** Only `str`, `int`, `float`, `bool` are accepted for `to: str`.

### Execution Model

1. Create a **working copy** of the input row
2. For each conversion in order:
   - Look up field in working copy
   - If field missing → error
   - If value is `None` → error
   - If value already target type → no-op (idempotent)
   - Otherwise apply conversion rules
   - Write result back to field
3. If **all conversions succeed**: emit working copy as success
4. If **any conversion fails**: emit **original input row** as error (no partial mutations)

### Contract Propagation

- No fields added or removed
- Field types become more specific in output contract if type tracking is enabled

### Audit Trail

```json
{
  "action": "coerced",
  "fields_coerced": ["price", "quantity", "is_active"],
  "fields_unchanged": ["user_id"],
  "rules_evaluated": 4
}
```

- `fields_coerced`: Fields where value was actually converted (in operation order)
- `fields_unchanged`: Fields where rule was evaluated but value was already target type
- `rules_evaluated`: Total conversion rules processed

**Note:** Audit lists preserve operation order rather than sorting alphabetically, for easier reconciliation with config.

### Example

**Input:**
```json
{"price": " 12.50 ", "quantity": "3", "is_active": "false", "user_id": 42}
```

**Config:**
```yaml
conversions:
  - field: price
    to: float
  - field: quantity
    to: int
  - field: is_active
    to: bool
  - field: user_id
    to: str
```

**Output:**
```json
{"price": 12.5, "quantity": 3, "is_active": false, "user_id": "42"}
```

## Error Metadata

When a row errors, the error metadata should identify the failing operation for debugging:

**`value_transform` error:**
```json
{
  "action": "error",
  "plugin": "value_transform",
  "target": "total",
  "expression": "row['price'] * row['quantity']",
  "reason": "Missing field 'quantity'"
}
```

**`type_coerce` error:**
```json
{
  "action": "error",
  "plugin": "type_coerce",
  "field": "is_active",
  "target_type": "bool",
  "reason": "Value 'maybe' is not a valid boolean"
}
```

## Typical Pipeline Pattern

```yaml
transforms:
  - plugin: type_coerce        # 1. Normalize types from source
    options:
      schema:
        mode: observed
      conversions:
        - field: price
          to: float
        - field: quantity
          to: int

  - plugin: value_transform    # 2. Compute with clean types
    options:
      schema:
        mode: observed
      operations:
        - target: subtotal
          expression: "row['price'] * row['quantity']"
        - target: tax
          expression: "row['subtotal'] * 0.2"
        - target: total
          expression: "row['subtotal'] + row['tax']"
```

## Out of Scope

The following are explicitly **not** included in V1:

- **DateTime parsing** — Complex, locale-dependent; separate plugin if needed
- **JSON string parsing** — Security implications; separate plugin if needed
- **Per-operation error handling** — Atomic row semantics only
- **Null policies** — `None` is always an error; no configurable null handling
- **Custom format strings** — No sprintf-style formatting

## Implementation Notes

### Shared Base Class

Both plugins share common patterns that could be extracted:

- Atomic row mutation with rollback on failure
- Ordered operation evaluation on working copy
- Audit trail with fields_modified/fields_added tracking
- Schema contract propagation

### File Locations

```
src/elspeth/plugins/transforms/
├── value_transform.py      # New
├── type_coerce.py          # New
├── field_mapper.py         # Existing (similar patterns)
└── truncate.py             # Existing (similar patterns)
```

### Test Coverage Requirements

- All conversion edge cases (especially bool mapping)
- Atomic failure semantics (partial mutation rollback)
- Sequential operation visibility
- Self-reference overwrite
- Duplicate target handling
- Schema contract propagation
- Expression security (inherited from ExpressionParser tests)

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Reuse ExpressionParser | Already audited, secure, consistent with gates |
| Separate type_coerce plugin | Keeps coercion explicit and auditable |
| No per-operation error handling | Row is atomic unit in ELSPETH |
| Strict bool conversion | Python truthiness creates silent data corruption |
| No int truncation | Silent data loss is unacceptable |
| Null → error | No fabrication principle |
| Trim whitespace for parsing | Practical without being ambiguous |
| Reject NaN/inf for float | Non-finite values cause downstream issues in JSON, DBs, aggregations |
| English-only bool strings | Prevents scope creep; localization is a V2 concern |
| No warning for duplicate targets | Sequential rewrites are legitimate; warnings would be noisy |
| Preserve operation order in audit | Easier to reconcile with config order for debugging |
