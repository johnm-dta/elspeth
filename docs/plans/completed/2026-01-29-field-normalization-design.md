# Field Normalization at Source Boundary

**Date:** 2026-01-29
**Status:** Approved (Revised after Review Board)
**Author:** Design collaboration (user + Claude)
**Reviewers:** Architecture, Python Engineering, QA, Systems Thinking

## Problem Statement

The schema contract feature (commit `be6a6a3`) assumes field names are valid Python identifiers (e.g., `customer_id`, `text`). Real-world CSV headers are often messy:

- `"CaSE Study1 !!!! xx!"` - special characters, mixed case
- `"User ID"` - spaces
- `"123_field"` - leading digits
- Missing headers entirely (headerless files)

This is a Tier 3 boundary problem: external data normalization belongs at the source.

## Design Goals

1. **Normalize messy headers** to valid Python identifiers automatically
2. **Support explicit overrides** when auto-normalization isn't enough
3. **Handle headerless files** with explicit column definitions
4. **Fail fast** on collisions, invalid mappings, or unparseable headers
5. **Full audit trail** of original-to-final field name resolution
6. **Validate template field references** against normalized names at plugin init

## Configuration Schema

New options for source plugins that read tabular external data:

```yaml
source:
  plugin: csv
  options:
    path: data.csv

    # For headerless files - ordered list of column names
    # Mutually exclusive with reading headers from file
    columns: [id, name, amount, category]

    # Auto-normalize messy headers to valid Python identifiers
    # Default: false (backwards compatible)
    normalize_fields: true

    # Override specific normalized names
    # Keys are NORMALIZED names (after auto-normalization), values are FINAL names
    field_mapping:
      case_study1_xx: cs1
      user_id: uid

    # CRITICAL: Schema field definitions MUST use FINAL names (post-normalization + mapping)
    # Schema validation occurs AFTER normalization, so declared fields must match final names.
    schema:
      mode: free
      fields:
        - "cs1: str"      # CORRECT - uses final name
        - "uid: int"      # CORRECT - uses final name
        # - "case_study1_xx: str"  # WRONG - intermediate name
        # - "User ID: str"         # WRONG - raw name
```

### Option Interactions

| `columns` | `normalize_fields` | `field_mapping` | Behavior |
|-----------|-------------------|-----------------|----------|
| Not set | `false` | Not set | Current behavior - raw headers used as-is |
| Not set | `true` | Not set | Headers auto-normalized |
| Not set | `true` | Set | Headers auto-normalized, then mapping applied |
| Not set | `false` | Set | Error: mapping without normalization is ambiguous |
| Set | N/A | Not set | Headerless mode - columns used directly |
| Set | `true` | N/A | Error: normalize_fields incompatible with columns |
| Set | N/A | Set | Headerless mode with mapping overrides allowed |

## Architecture

### Code Organization

**Normalization logic lives in a dedicated utility module, separate from config:**

```
src/elspeth/plugins/sources/
├── field_normalization.py    # NEW: Algorithm and resolution logic
├── csv_source.py             # Uses TabularSourceDataConfig + normalization utils
├── json_source.py            # Uses TabularSourceDataConfig + normalization utils
└── ...

src/elspeth/plugins/
├── config_base.py            # TabularSourceDataConfig (inheritance, not mixin)
└── ...
```

### Config Class Hierarchy

Use **inheritance** (not mixin) for tabular source config:

```python
# config_base.py
class TabularSourceDataConfig(SourceDataConfig):
    """Config for sources that read tabular external data with headers."""

    columns: list[str] | None = None
    normalize_fields: bool = False
    field_mapping: dict[str, str] | None = None

    @model_validator(mode="after")
    def _validate_normalization_options(self) -> Self:
        """Validate field normalization option interactions."""
        if self.columns is not None and self.normalize_fields:
            raise ValueError(
                "normalize_fields cannot be used with columns config. "
                "The columns config already provides clean names."
            )

        if self.field_mapping is not None and not self.normalize_fields and self.columns is None:
            raise ValueError(
                "field_mapping requires normalize_fields: true or columns config"
            )

        # Validate columns entries are valid identifiers and not keywords
        if self.columns is not None:
            _validate_field_names(self.columns, "columns")

        # Validate field_mapping values are valid identifiers and not keywords
        if self.field_mapping is not None:
            _validate_field_names(list(self.field_mapping.values()), "field_mapping values")

        return self
```

**Rationale:** Mixins create MRO complexity and conflate config with processing. Inheritance is cleaner for adding config fields. Processing logic stays in the utility module.

## Normalization Algorithm

Located in `src/elspeth/plugins/sources/field_normalization.py`:

```python
import keyword
import re
import unicodedata

# Pre-compiled regex patterns (module level for efficiency)
_NON_IDENTIFIER_CHARS = re.compile(r'[^\w]+')
_CONSECUTIVE_UNDERSCORES = re.compile(r'_+')


def normalize_field_name(raw: str) -> str:
    """Normalize messy header to valid Python identifier.

    Rules applied in order:
    1. Unicode NFC normalization (canonical composition)
    2. Strip leading/trailing whitespace
    3. Lowercase
    4. Replace non-identifier chars (not letter/digit/underscore) with underscore
    5. Collapse consecutive underscores to single underscore
    6. Strip leading/trailing underscores
    7. If result starts with digit, prefix with underscore
    8. If result is a Python keyword, append underscore
    9. If result is empty, raise error (header is unparseable)

    Args:
        raw: Original messy header name

    Returns:
        Valid Python identifier

    Raises:
        ValueError: If header normalizes to empty string
    """
    # Step 1: Unicode NFC normalization (handles combining characters)
    normalized = unicodedata.normalize('NFC', raw)

    # Step 2: Strip whitespace
    normalized = normalized.strip()

    # Step 3: Lowercase
    normalized = normalized.lower()

    # Step 4: Replace non-identifier chars with underscore
    normalized = _NON_IDENTIFIER_CHARS.sub('_', normalized)

    # Step 5: Collapse consecutive underscores
    normalized = _CONSECUTIVE_UNDERSCORES.sub('_', normalized)

    # Step 6: Strip leading/trailing underscores
    normalized = normalized.strip('_')

    # Step 7: Prefix if starts with digit
    if normalized and normalized[0].isdigit():
        normalized = f'_{normalized}'

    # Step 8: Handle Python keywords
    if keyword.iskeyword(normalized):
        normalized = f'{normalized}_'

    # Step 9: Validate non-empty result
    if not normalized:
        raise ValueError(f"Header '{raw}' normalizes to empty string")

    # Defense-in-depth: verify result is valid identifier
    if not normalized.isidentifier():
        raise ValueError(
            f"Header '{raw}' normalized to '{normalized}' which is not a valid identifier. "
            f"This is a bug in the normalization algorithm."
        )

    return normalized
```

### Examples

| Raw Header | Normalized | Notes |
|------------|------------|-------|
| `"CaSE Study1 !!!! xx!"` | `case_study1_xx` | Spaces and `!` → `_`, collapsed |
| `"User ID"` | `user_id` | Space → `_` |
| `"123_field"` | `_123_field` | Leading digit gets `_` prefix |
| `"data.field"` | `data_field` | Dot → `_` |
| `"  Amount  "` | `amount` | Whitespace stripped |
| `"class"` | `class_` | Python keyword gets `_` suffix |
| `"for"` | `for_` | Python keyword gets `_` suffix |
| `"café"` | `café` | Valid identifier (PEP 3131) |
| `"Status 🔥"` | `status` | Emoji stripped |
| `"\ufeffid"` | `id` | BOM stripped |
| `"id\u200b"` | `id` | Zero-width chars stripped |
| `"!!!"` | **ERROR** | Nothing left after normalization |

### Field Resolution Function

```python
@dataclass
class FieldResolution:
    """Result of field name resolution."""
    final_headers: list[str]
    resolution_mapping: dict[str, str]  # raw_name → final_name


def resolve_field_names(
    raw_headers: list[str] | None,
    config: TabularSourceDataConfig,
) -> FieldResolution:
    """Resolve final field names from raw headers and config.

    Args:
        raw_headers: Headers from file, or None if using columns config
        config: Source configuration with normalization options

    Returns:
        FieldResolution with final headers and audit mapping

    Raises:
        ValueError: On collision, invalid mapping, or empty normalization
    """
    # ... implementation
```

## Processing Flow

### Header Resolution Timing

**CRITICAL:** Header resolution happens at the **start of `load()`**, not in `__init__`.

**Rationale:** File may not exist or be accessible at config validation time (e.g., path resolved later, file generated by prior step). The "fail fast" guarantee means bad headers crash **before any data rows are processed**, not at config time.

### Load Phase Flow

```
START OF load():
─────────────────────────────────────────────────────────────────

1. Determine raw headers:
   IF columns provided:
       raw_headers = None (headerless mode)
       effective_headers = config.columns
   ELSE:
       Read first row as raw_headers
       IF file appears empty or malformed:
           FAIL: "CSV file has no header row. Provide 'columns' config"

2. Normalize (if enabled):
   IF normalize_fields AND raw_headers is not None:
       normalized_headers = [normalize_field_name(h) for h in raw_headers]

       # Check for collisions AFTER normalization
       _check_collisions(raw_headers, normalized_headers)

       effective_headers = normalized_headers
   ELSE:
       effective_headers = raw_headers or config.columns

3. Apply field mapping (if provided):
   IF field_mapping:
       # Validate all mapping keys exist
       missing_keys = set(field_mapping.keys()) - set(effective_headers)
       IF missing_keys:
           FAIL with available headers listed

       # Apply mapping
       final_headers = [field_mapping.get(h, h) for h in effective_headers]

       # Check for collisions AFTER mapping (e.g., {a: x, b: x})
       _check_post_mapping_collisions(effective_headers, final_headers, field_mapping)
   ELSE:
       final_headers = effective_headers

4. Build resolution mapping for audit:
   IF raw_headers is not None:
       resolution_mapping = {raw: final for raw, final in zip(raw_headers, final_headers)}
   ELSE:
       resolution_mapping = {col: final for col, final in zip(config.columns, final_headers)}

5. Store resolution for audit trail (in source instance)

6. Validate final headers against schema

ROW PROCESSING (per row):
─────────────────────────────────────────────────────────────────

1. Read values from file
2. Zip with final_headers (pre-computed above)
3. Validate against schema
4. Yield SourceRow

All normalization overhead is before first row. Zero per-row cost.
```

## Collision Detection

### After Normalization

Detect when multiple raw headers normalize to the same value:

```python
def _check_collisions(raw_headers: list[str], normalized_headers: list[str]) -> None:
    """Check for collisions after normalization.

    Raises:
        ValueError: With ALL colliding headers and their positions
    """
    seen: dict[str, list[tuple[int, str]]] = {}  # normalized → [(position, raw), ...]

    for i, (raw, norm) in enumerate(zip(raw_headers, normalized_headers)):
        seen.setdefault(norm, []).append((i, raw))

    collisions = {norm: sources for norm, sources in seen.items() if len(sources) > 1}

    if collisions:
        details = []
        for norm, sources in sorted(collisions.items()):
            source_desc = ", ".join(f"column {i} ('{raw}')" for i, raw in sources)
            details.append(f"  '{norm}' ← {source_desc}")

        raise ValueError(
            f"Field name collision after normalization:\n" + "\n".join(details)
        )
```

### After Mapping

Detect when field_mapping creates collisions:

```python
def _check_post_mapping_collisions(
    pre_mapping: list[str],
    post_mapping: list[str],
    field_mapping: dict[str, str],
) -> None:
    """Check for collisions created by field_mapping.

    Example: {a: x, b: x} maps two fields to same name.
    """
    if len(post_mapping) != len(set(post_mapping)):
        # Find which mapping entries caused collision
        target_counts: dict[str, list[str]] = {}
        for source, target in field_mapping.items():
            target_counts.setdefault(target, []).append(source)

        collisions = {t: s for t, s in target_counts.items() if len(s) > 1}

        if collisions:
            details = [f"  '{t}' ← {', '.join(repr(s) for s in sources)}"
                       for t, sources in sorted(collisions.items())]
            raise ValueError(
                f"field_mapping creates collision:\n" + "\n".join(details)
            )
```

## Error Handling

All failures are loud and early:

| Scenario | When Detected | Error Message |
|----------|---------------|---------------|
| Collision after normalization | Load start | `"Field collision: column 3 ('Case Study 1') and column 7 ('case-study-1') both normalize to 'case_study_1'"` |
| Multi-way collision | Load start | Lists ALL colliding headers with positions |
| Header normalizes to empty | Load start | `"Header '!!!' at column 3 normalizes to empty string"` |
| Mapping key not found | Load start | `"field_mapping key 'foo' not found. Available: ['case_study1_xx', 'user_id', ...]"` |
| Mapping creates collision | Load start | `"field_mapping creates collision: 'x' ← 'a', 'b'"` |
| Mapping value is keyword | Config validation | `"field_mapping value 'class' is a Python keyword"` |
| Mapping value invalid identifier | Config validation | `"field_mapping value '123' is not a valid identifier"` |
| Headerless without `columns` | Load start | `"CSV has no header row. Provide 'columns' config"` |
| `columns` count mismatch | First row | `"columns config has 4 fields but CSV row has 5 values"` |
| `columns` count under | First row | `"columns config has 4 fields but CSV row has 3 values"` |
| Duplicate in `columns` | Config validation | `"Duplicate field name 'id' in columns config"` |
| `columns` entry is keyword | Config validation | `"columns entry 'class' is a Python keyword"` |
| `columns` entry invalid | Config validation | `"columns entry '123' is not a valid identifier"` |
| `columns` + `normalize_fields` | Config validation | `"normalize_fields cannot be used with columns"` |
| `field_mapping` without context | Config validation | `"field_mapping requires normalize_fields: true or columns config"` |
| Empty CSV file | Load start | `"CSV file is empty"` |

## Template Field Validation

**CRITICAL:** Transforms using Jinja2 templates must validate field references against the schema at plugin init.

### The Problem

Templates reference fields by name. If normalization changes `User ID` → `user_id`, but a template uses `{{ row.User_ID }}`, Jinja2 silently returns empty string. This creates audit trail corruption.

### The Solution

At transform plugin init, validate template field references:

```python
# In transform plugin __init__() after schema is determined:
from elspeth.core.templates import extract_jinja2_fields

template_fields = extract_jinja2_fields(self._template_string)
schema_fields = frozenset(self._schema_class.model_fields.keys())

missing = template_fields - schema_fields
if missing:
    raise ValueError(
        f"Template references fields not in schema: {sorted(missing)}\n"
        f"Available fields: {sorted(schema_fields)}\n"
        f"Hint: If source uses normalize_fields, field names are transformed. "
        f"Check source's field_resolution in audit trail for actual names."
    )
```

### Why This Matters

- **Without validation:** Silent empty strings in output, audit trail records garbage
- **With validation:** Immediate failure at pipeline construction, clear error message

## Audit Trail

### Storage Mechanism

The `field_resolution` mapping is added to the source's config dict passed to `recorder.register_node()`. This keeps field resolution visible in the audit trail without schema changes.

```python
# In source plugin, after resolution:
self._field_resolution = resolution.resolution_mapping

# The orchestrator includes this when registering:
# config dict already contains plugin options, field_resolution is added
```

### Recorded Data

Recorded once per run in source node metadata:

```python
{
    "source_plugin": "csv",
    "path": "data.csv",
    "normalize_fields": true,
    "field_resolution": {
        # Complete lineage: raw header → final field name
        "CaSE Study1 !!!! xx!": "cs1",
        "User ID": "uid",
        "Amount": "amount",  # auto-norm only, no override
    },
    # Original config preserved (field_mapping is part of options)
    "schema_config": { ... }
}
```

**Note:** `field_mapping_applied` removed per review - it's redundant with the original config and `field_resolution` derivable from the difference.

### Querying

Auditor asking "where did field `cs1` come from?":
1. Query source node metadata for run
2. Look up `cs1` in `field_resolution` (reverse lookup)
3. Find original header `"CaSE Study1 !!!! xx!"`

## Implementation Scope

### In Scope

**New utility module:**
- `src/elspeth/plugins/sources/field_normalization.py` - Algorithm and resolution logic

**Config extension:**
- `src/elspeth/plugins/config_base.py` - `TabularSourceDataConfig` class

**Source plugins:**
- `src/elspeth/plugins/sources/csv_source.py`
- `src/elspeth/plugins/sources/json_source.py`
- `src/elspeth/plugins/azure/blob_source.py`

**Template validation (transforms):**
- Update LLM transform base classes to validate template fields at init

**Tests:**
- Unit tests for normalization algorithm (including Unicode, keywords)
- Unit tests for collision detection
- Integration tests for each source plugin
- Integration tests for template field validation
- Error case coverage

### Out of Scope (YAGNI)

| Not Doing | Why |
|-----------|-----|
| Regex-based mapping rules | Complexity explosion; explicit mapping is clearer |
| Per-column type coercion in mapping | Schema already handles type coercion |
| Fuzzy/approximate header matching | Too magic; prefer explicit `columns` for headerless |
| Reverse mapping at sinks | Sinks output clean names; use transform if messy output needed |
| Runtime mapping changes | Config is immutable per run |
| Max field name length | YAGNI unless we hit real issues |

## Test Requirements

### P0 - Must Have (Blocking)

**Unit Tests (Normalization Algorithm):**
1. Basic normalization (design examples)
2. Unicode BOM stripping (`"\ufeffid"` → `"id"`)
3. Zero-width character removal (`"id\u200b"` → `"id"`)
4. Python keyword detection (`"class"` → `"class_"`)
5. Multi-way collision detection (3+ headers → same value)
6. Empty result detection (`"!!!"` → error)
7. Emoji stripping (`"Status 🔥"` → `"status"`)

**Integration Tests (CSVSource):**
8. Empty CSV file → error with clear message
9. Headerless mode with keyword in `columns` → config validation error
10. Collision after `field_mapping` → error at init
11. `field_mapping` to Python keyword → config validation error
12. Column count mismatch (both over and under) → error on first row
13. Normalized headers with schema contracts → DAG validation passes
14. Template field validation catches missing normalized fields

### P1 - Should Have

15. CJK character normalization (PEP 3131 compliance)
16. Accented characters with NFC normalization
17. Header-only CSV (0 data rows) succeeds
18. Single-column CSV
19. Field mapping key as substring of another field
20. Audit trail contains complete `field_resolution` mapping

### P2 - Nice to Have

21. RTL text normalization
22. Very long header names (1000+ chars)
23. Performance test: 1000-column CSV normalization

## Backwards Compatibility

- Default `normalize_fields: false` preserves current behavior
- Existing pipelines with clean headers continue working unchanged
- New options are purely additive

## Warning: Field Mapping Accumulation

Large `field_mapping` sections are a code smell indicating data quality issues upstream.

**Recommendation:** If `field_mapping` grows beyond ~10 entries, consider:
1. Cleaning headers at data source
2. Using `columns` config for complete control
3. Creating a dedicated header-cleaning transform upstream

## Success Criteria

1. Pipeline with `normalize_fields: true` transforms `"CaSE Study1 !!!! xx!"` → `case_study1_xx`
2. Python keywords get `_` suffix (`"class"` → `"class_"`)
3. Field mapping `{case_study1_xx: cs1}` produces final name `cs1`
4. Headerless file with `columns: [a, b, c]` works correctly
5. Collision between two headers that normalize identically fails at load start
6. Collision after field_mapping fails at load start
7. Invalid mapping key fails at load start with helpful error
8. Invalid `columns` entries fail at config validation
9. Audit trail shows complete `field_resolution` mapping
10. Schema contract validation works with normalized/mapped field names
11. Template field validation catches references to pre-normalization names

## Review Board Sign-off

| Reviewer | Verdict | Key Concerns Addressed |
|----------|---------|----------------------|
| Architecture | Approved | Mixin → inheritance, utility module separation |
| Python Engineering | Approved | Compiled regex, Pydantic validators, collision detection |
| Quality Assurance | Approved | Unicode handling, keywords, comprehensive test list |
| Systems Thinking | Approved | Template validation, schema field name documentation |
