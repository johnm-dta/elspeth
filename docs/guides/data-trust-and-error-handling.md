# Data Trust Model & Error Handling Guide

**MANDATORY READING** when working on: Sources, Transforms, Sinks, Gates, Aggregations, external API integrations, Landscape recording, or any code that handles data crossing system boundaries.

## Overview

ELSPETH's error handling philosophy is built on a fundamental insight: **different data has different trust levels**, and the correct error response depends entirely on whose data failed.

This guide covers:
1. The Three-Tier Trust Model
2. Plugin Ownership implications
3. The Defensive Programming Prohibition
4. Decision frameworks for error handling

## The Three-Tier Trust Model

ELSPETH has three fundamentally different trust tiers with distinct handling rules:

### Tier 1: Our Data (Audit Database / Landscape) - FULL TRUST

**Must be 100% pristine at all times.** We wrote it, we own it, we trust it completely.

- Bad data in the audit trail = **crash immediately**
- No coercion, no defaults, no silent recovery
- If we read garbage from our own database, something catastrophic happened (bug in our code, database corruption, tampering)
- Every field must be exactly what we expect - wrong type = crash, NULL where unexpected = crash, invalid enum value = crash

**Why this matters for audit integrity:**

The audit trail is the legal record. Silently coercing bad data is evidence tampering. If an auditor asks "why did row 42 get routed here?" and we give a confident wrong answer because we coerced garbage into a valid-looking value, we've committed fraud.

**Examples of Tier 1 data:**
- `landscape.get_row_state(token_id)` results
- `node_states` table records
- `calls` table records
- Run configuration stored in `runs` table
- Any data read from the audit database

**Correct handling:**
```python
# CORRECT - Let it crash if our data is corrupt
row_state = landscape.get_row_state(token_id)
node_id = row_state.node_id  # Direct access - crash if missing

# WRONG - Defensive handling on our own data
row_state = landscape.get_row_state(token_id)
node_id = getattr(row_state, 'node_id', None)  # NO! Hides corruption
if node_id is None:
    node_id = "unknown"  # NO! Evidence tampering
```

### Tier 2: Pipeline Data (Post-Source) - ELEVATED TRUST ("Probably OK")

**Type-valid but potentially operation-unsafe.** Data that passed source validation.

- Types are trustworthy (source validated and/or coerced them)
- Values might still cause operation failures (division by zero, invalid date formats, etc.)
- Transforms/sinks **expect conformance** - if types are wrong, that's an upstream plugin bug
- **No coercion** at transform/sink level - if a transform receives `"42"` when it expected `int`, that's a bug in the source or upstream transform

**Why plugins don't coerce:**

Plugins have contractual obligations. If a transform's `output_schema` says `int` and it outputs `str`, that's a bug we fix by fixing the plugin, not by coercing downstream. Coercing downstream would:
1. Hide the upstream bug
2. Make debugging impossible ("where did this become a string?")
3. Potentially produce wrong results silently

**Critical nuance: Type-safe doesn't mean operation-safe:**

```python
# Data is type-valid (int), but operation fails
row = {"divisor": 0}  # Passed source validation ✓
result = 100 / row["divisor"]  # 💥 ZeroDivisionError - wrap this!

# Data is type-valid (str), but content is problematic
row = {"date": "not-a-date"}  # Passed as str ✓
parsed = datetime.fromisoformat(row["date"])  # 💥 ValueError - wrap this!

# Data is type-valid (list), but might be empty
row = {"items": []}  # Passed source validation ✓
first = row["items"][0]  # 💥 IndexError - wrap this!
```

**Correct handling:**
```python
def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    # Wrap operations that can fail on valid-typed but problematic values
    try:
        result = row["numerator"] / row["denominator"]
    except ZeroDivisionError:
        return TransformResult.error({"reason": "division_by_zero", "row_id": ctx.row_id})

    return TransformResult.success({"result": result})
```

### Tier 3: External Data (Source Input) - ZERO TRUST

**Can be literal trash.** We don't control what external systems feed us.

- Malformed CSV rows, NULLs everywhere, wrong types, unexpected JSON structures
- **Validate at the boundary, coerce where possible, record what we got**
- Sources MAY coerce: `"42"` → `42`, `"true"` → `True` (normalizing external data)
- Quarantine rows that can't be coerced/validated
- The audit trail records "row 42 was quarantined because field X was NULL" - that's a valid audit outcome

**Why user data is different:**

User data is a trust boundary. A CSV with garbage in row 500 shouldn't crash the entire pipeline - we record the problem, quarantine the row, and keep processing the other 10,000 rows. The audit trail documents exactly what happened.

**Examples of Tier 3 data:**
- CSV file contents
- JSON API responses
- Database query results from external databases
- LLM API responses
- HTTP webhook payloads
- Message queue messages
- File contents read by transforms

### The Trust Flow Diagram

```text
EXTERNAL DATA              PIPELINE DATA              AUDIT TRAIL
(zero trust)               (elevated trust)           (full trust)
                           "probably ok"

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│ External Source │        │ Transform/Sink  │        │ Landscape DB    │
│                 │        │                 │        │                 │
│ • Coerce OK     │───────►│ • No coercion   │───────►│ • Crash on      │
│ • Validate      │ types  │ • Expect types  │ record │   any anomaly   │
│ • Quarantine    │ valid  │ • Wrap ops on   │ what   │ • No coercion   │
│   failures      │        │   row values    │ we     │   ever          │
│                 │        │ • Bug if types  │ saw    │                 │
│                 │        │   are wrong     │        │                 │
└─────────────────┘        └─────────────────┘        └─────────────────┘
         │                          │
         │                          │
    Source is the              Operations on row
    ONLY place coercion        values need wrapping
    is allowed                 (values can still fail)
```

## External Call Boundaries in Transforms

**CRITICAL:** Trust tiers are about **data flows**, not plugin types. **Any data crossing from an external system is Tier 3**, regardless of which plugin makes the call.

Transforms that make external calls (LLM APIs, HTTP requests, database queries) create **mini Tier 3 boundaries** within their implementation. The transform receives Tier 2 data (the row), but the external response is Tier 3.

### The Pattern for External Calls

```python
def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    # row enters as Tier 2 (pipeline data - trust the schema)
    customer_id = row["customer_id"]  # Direct access OK - it's Tier 2

    # External call creates Tier 3 boundary
    try:
        llm_response = self._llm_client.query(prompt)  # EXTERNAL DATA - zero trust
    except Exception as e:
        return TransformResult.error({"reason": "llm_call_failed", "error": str(e)})

    # IMMEDIATELY validate at the boundary - don't let "their data" travel
    try:
        parsed = json.loads(llm_response.content)
    except json.JSONDecodeError:
        return TransformResult.error({"reason": "invalid_json", "raw": llm_response.content[:200]})

    # Validate structure type IMMEDIATELY
    if not isinstance(parsed, dict):
        return TransformResult.error({
            "reason": "invalid_json_type",
            "expected": "object",
            "actual": type(parsed).__name__
        })

    # Validate required fields exist
    if "category" not in parsed:
        return TransformResult.error({
            "reason": "missing_field",
            "field": "category",
            "received_keys": list(parsed.keys())
        })

    # NOW it's our data (Tier 2) - add to row and continue
    row["llm_classification"] = parsed["category"]  # Safe - validated above
    return TransformResult.success(row)
```

### The Rule: Minimize Distance Before Validation

- ✅ **Validate immediately** - right after the external call returns
- ✅ **Coerce once** - normalize types at the boundary
- ✅ **Trust thereafter** - once validated, it's Tier 2 pipeline data
- ❌ **Don't carry raw external data** - passing `llm_response` to helper methods without validation
- ❌ **Don't defer validation** - "I'll check it later when I use it"
- ❌ **Don't validate multiple times** - if it's validated once, trust it

### Common External Boundaries in Transforms

| External Call Type | Tier 3 Boundary | Validation Pattern |
|-------------------|-----------------|-------------------|
| LLM API response | Response content | Wrap JSON parse, validate type is dict, check required fields |
| HTTP API response | Response body | Wrap request, validate status code, parse and validate schema |
| Database query results | Result rows | Validate row structure, handle missing fields, coerce types |
| File reads (in transform) | File contents | Same validation as source plugins |
| Message queue consume | Message payload | Parse format, validate schema, quarantine malformed messages |

### Real Example from azure_multi_query_llm.py

This is the correct pattern as implemented in production:

```python
# Line 227-236: External call (Tier 3 boundary created)
try:
    response = await self._llm_executor.execute_llm_call(...)
except Exception as e:
    return TransformResult.error(...)  # Wrapped immediately

# Line 241-251: IMMEDIATE validation at boundary
try:
    parsed = json.loads(response.content)
except json.JSONDecodeError:
    return TransformResult.error(...)  # Can't parse - reject immediately

# Line 253-263: Structure type validation (defense against non-dict JSON)
if not isinstance(parsed, dict):
    return TransformResult.error({
        "reason": "invalid_json_type",
        "expected": "object",
        "actual": type(parsed).__name__
    })

# Line 266-274: NOW safe to use - it's validated Tier 2 data
output[output_key] = parsed[json_field]  # No defensive .get() needed
```

From this point forward, `parsed` is treated as Tier 2 pipeline data. No more validation. No `.get()` calls. We trust it because we validated it at the boundary.

## Coercion Rules by Plugin Type

| Plugin Type | Coercion Allowed? | Rationale |
|-------------|-------------------|-----------|
| **Source** | ✅ Yes | Normalizes external data at ingestion boundary |
| **Transform (on row data)** | ❌ No | Receives validated data; wrong types = upstream bug |
| **Transform (on external call response)** | ✅ Yes | External response is Tier 3 - validate/coerce immediately |
| **Sink** | ❌ No | Receives validated data; wrong types = upstream bug |
| **Gate** | ❌ No | Receives validated data; wrong types = upstream bug |
| **Aggregation** | ❌ No | Receives validated data; wrong types = upstream bug |

## Operation Wrapping Rules

| What You're Accessing | Wrap in try/except? | Why |
|----------------------|---------------------|-----|
| `self._config.field` | ❌ No | Our code, our config - crash on bug |
| `self._internal_state` | ❌ No | Our code - crash on bug |
| `landscape.get_row_state(token_id)` | ❌ No | Our data - crash on corruption |
| `row["field"]` arithmetic/parsing | ✅ Yes | Their data values can fail operations |
| `external_api.call(row["id"])` | ✅ Yes | External system, anything can happen |
| `json.loads(external_response)` | ✅ Yes | External data - validate immediately |
| `validated_dict["field"]` | ❌ No | Already validated at boundary - trust it |

**Rule of thumb:**

- **Reading from Landscape tables?** Crash on any anomaly - it's our data.
- **Operating on row field values?** Wrap operations, return error result, quarantine row.
- **Calling external systems?** Wrap call AND validate response immediately at boundary.
- **Using already-validated external data?** Trust it - no defensive `.get()` needed.
- **Accessing internal state?** Let it crash - that's a bug to fix.

## Plugin Ownership: System Code, Not User Code

**CRITICAL DISTINCTION:** All plugins (Sources, Transforms, Gates, Aggregations, Sinks) are **system-owned code**, not user-provided extensions.

### What This Means

```text
┌─────────────────────────────────────────────────────────────────┐
│                     SYSTEM-OWNED (Full Trust)                    │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Sources    │  │  Transforms  │  │    Sinks     │           │
│  │  (CSVSource, │  │ (FieldMapper,│  │  (CSVSink,   │           │
│  │   APISource) │  │  LLMTransform)│  │   DBSink)    │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │    Engine    │  │  Landscape   │  │   Contracts  │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ processes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     USER-OWNED (Zero Trust)                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                      USER DATA                            │   │
│  │   CSV files, API responses, database rows, LLM outputs    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Implications for Error Handling

| Scenario | Correct Response | WRONG Response |
|----------|------------------|----------------|
| Plugin method throws exception | **CRASH** - bug in our code | Catch and log silently |
| Plugin returns wrong type | **CRASH** - bug in our code | Coerce to expected type |
| Plugin missing expected attribute | **CRASH** - interface violation | Use `getattr(x, 'attr', default)` |
| User data has wrong type | Quarantine row, continue | Crash the pipeline |
| User data missing field | Quarantine row, continue | Crash the pipeline |

### Why This Matters for Audit Integrity

A defective plugin that silently produces wrong results is **worse than a crash**:

1. **Crash:** Pipeline stops, operator investigates, bug gets fixed
2. **Silent wrong result:** Data flows through, gets recorded as "correct," auditors see garbage, trust is destroyed

**Example of the problem:**

```python
# WRONG - hides plugin bugs, destroys audit integrity
try:
    result = transform.process(row, ctx)
except Exception:
    result = row  # "just pass through on error"
    logger.warning("Transform failed, using original row")

# RIGHT - plugin bugs crash immediately
result = transform.process(row, ctx)  # Let it crash
```

If `transform.process()` has a bug, we MUST know about it. Silently passing through the original row means the audit trail now contains data that "looks processed" but wasn't - this is evidence tampering.

### NOT a Plugin Marketplace

ELSPETH uses `pluggy` for clean architecture (hooks, extensibility), NOT to accept arbitrary user plugins:

- Plugins are developed, tested, and deployed as part of ELSPETH
- Plugin code is reviewed with the same rigor as engine code
- Plugin bugs are system bugs - they get fixed in the codebase
- Users configure which plugins to use, they don't write their own

If a future version supports user-authored plugins, those would be sandboxed and treated as untrusted - but that's not the current architecture.

## The Defensive Programming Prohibition

### The Core Rule

This codebase prohibits defensive patterns that mask bugs instead of fixing them. Do not use `.get()`, `getattr()`, `hasattr()`, `isinstance()`, or silent exception handling to suppress errors from:
- Nonexistent attributes
- Malformed data from our own code
- Incorrect types in our own data structures

### The Common Anti-Pattern

A common anti-pattern is when an LLM hallucinates a variable or field name, the code fails, and the "fix" is wrapping it in `getattr(obj, "hallucinated_field", None)` to silence the error. This hides the real bug.

```python
# The bug: LLM hallucinated "user_name" but the field is "username"
user = get_user(id)
name = user.user_name  # AttributeError!

# WRONG "fix" - hides the bug
name = getattr(user, "user_name", "Unknown")  # Silently returns "Unknown"

# RIGHT fix - correct the field name
name = user.username  # Use the actual field
```

### When Code Fails, Fix the Actual Cause

- Correct the field name
- Migrate the data source to emit proper types
- Fix the broken integration
- Update the schema
- Fix the upstream plugin

Typed dataclasses with discriminator fields serve as contracts; access fields directly (`obj.field`) not defensively (`obj.get("field")`). If code would fail without a defensive pattern, that failure is a bug to fix, not a symptom to suppress.

### Legitimate Uses of Defensive Patterns

This prohibition does not extend to genuine use cases where defensive handling is necessary:

#### 1. Operations on Row Values (Their Data)

Even type-valid row data can cause operation failures. Wrap these operations:

```python
# CORRECT - wrapping operations on their data
def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    try:
        result = row["numerator"] / row["denominator"]  # Their data can be 0
    except ZeroDivisionError:
        return TransformResult.error({"reason": "division_by_zero"})
    return TransformResult.success({"result": result})

# WRONG - wrapping access to OUR internal state
def process(self, row: dict[str, Any], ctx: PluginContext) -> TransformResult:
    try:
        batch_avg = self._total / self._batch_count  # OUR bug if _batch_count is 0
    except ZeroDivisionError:
        batch_avg = 0  # NO! This hides our initialization bug
```

**The distinction:** Wrapping `row["x"] / row["y"]` is correct because `row` is their data. Wrapping `self._x / self._y` is wrong because `self` is our code.

#### 2. External System Boundaries

- **External API responses**: Validating JSON structure from LLM providers or HTTP endpoints before processing
- **Source plugin input**: Coercing/validating external data at ingestion (see Tier 3 above)

```python
# CORRECT - external API response is Tier 3
try:
    response = requests.get(url)
    data = response.json()
except (requests.RequestException, json.JSONDecodeError) as e:
    return TransformResult.error({"reason": "api_call_failed", "error": str(e)})

# Validate structure before trusting
if not isinstance(data, dict) or "result" not in data:
    return TransformResult.error({"reason": "invalid_response_structure"})
```

#### 3. Framework Boundaries

- **Plugin schema contracts**: Type checking at plugin boundaries where external code meets the framework
- **Configuration validation**: Pydantic validators rejecting malformed config at load time

#### 4. Serialization

- **Pandas dtype normalization**: Converting `numpy.int64` → `int` in canonicalization
- **Serialization polymorphism**: Handling `datetime`, `Decimal`, `bytes` in canonical JSON

```python
# CORRECT - serialization needs type handling
def normalize_for_json(value):
    if isinstance(value, numpy.integer):
        return int(value)
    if isinstance(value, numpy.floating):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value
```

### The Decision Test

Ask yourself these questions:

| Question | If Yes | If No |
|----------|--------|-------|
| Is this protecting against user-provided data values? | ✅ Wrap it | — |
| Is this at an external system boundary (API, file, DB)? | ✅ Wrap it | — |
| Would this fail due to a bug in code we control? | — | ❌ Let it crash |
| Am I adding this because "something might be None"? | — | ❌ Fix the root cause |

**If you're wrapping to hide a bug that "shouldn't happen," remove the wrapper and fix the bug.**

### Quick Reference: Wrap or Crash?

| Code Pattern | Wrap? | Reasoning |
|--------------|-------|-----------|
| `row["amount"] / row["quantity"]` | ✅ Yes | Their data values |
| `self._buffer / self._count` | ❌ No | Our initialization bug |
| `json.loads(api_response.text)` | ✅ Yes | External data |
| `json.loads(self._cached_config)` | ❌ No | Our cached data |
| `user_input.strip().lower()` | ✅ Yes | User input can be weird |
| `self._state.current_phase.name` | ❌ No | Our state machine |
| `external_db.query(sql)` | ✅ Yes | External system |
| `landscape.get_row(id)` | ❌ No | Our audit data |
| `row.get("optional_field")` | ⚠️ Depends | Only if schema says optional |
| `getattr(plugin, "process")` | ❌ No | Plugin interface contract |

## Summary: The Mental Model

1. **Whose data is it?**
   - Our data (Tier 1): Crash on anomaly
   - Pipeline data (Tier 2): Trust types, wrap value operations
   - External data (Tier 3): Validate everything at boundary

2. **Where is the boundary?**
   - Source plugins: Tier 3 → Tier 2 (validate, coerce, quarantine)
   - Transform external calls: Tier 3 boundary within Tier 2 context
   - Landscape writes: Tier 2 → Tier 1 (we control this, crash on bugs)

3. **What's the failure mode?**
   - User data problem: Quarantine row, continue pipeline, audit records problem
   - Our bug: Crash immediately, fix the code
   - External system down: Return error, retry policy handles it

4. **Is this defensive pattern hiding a bug?**
   - If removing the pattern would reveal a bug in our code: Remove it, fix the bug
   - If removing the pattern would crash on valid external data: Keep it
