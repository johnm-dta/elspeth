---
name: tier-model-deep-dive
description: >
  Detailed trust tier examples and rules for all ELSPETH components — pipeline plugins
  (external call boundaries, coercion rules, operation wrapping, template error handling),
  web server (request validation, secret handling, config trust), and the fabrication
  decision test. Use when writing or modifying code that handles data at trust boundaries.
---

# Three-Tier Trust Model — Deep Reference

> **Model complexity warning:** This skill requires careful judgment about trust
> boundaries, coercion vs fabrication, and when to wrap vs crash. Do not delegate
> trust-boundary work to fast/simple models (Haiku, etc.) — they pattern-match on
> surface syntax and will produce plausible-looking code that violates the trust
> model in subtle ways (e.g. adding `.get()` at Tier 2, swallowing exceptions at
> Tier 1, inferring values at Tier 3). If a subagent must touch trust-boundary
> code, use Sonnet or Opus and include this skill in the prompt.

This skill provides detailed code examples, tables, and boundary rules for ELSPETH's
three-tier trust model. For the core rules (what each tier means), see CLAUDE.md.

## Quick Decision Table (Mechanical — No Judgment Required)

For every line of code that accesses data, match it against this table top-to-bottom.
First match wins. If no match, ask for help — do not guess.

### What am I reading?

| Pattern | Tier | Action |
|---------|------|--------|
| `landscape.*`, `recorder.*`, `query_repository.*` | T1 | Direct access. No try/except. No `.get()`. Crash on anomaly. |
| `checkpoint_data[...]`, `json.loads(stored_audit_json)` | T1 | Direct access. Crash on anomaly. Serialization doesn't change tier. |
| `app.state.config.*`, `app.state.settings.*`, `self._config.*` | T2 | Direct access. No `getattr()` defaults. Crash on bug. |
| `row["field"]` in a transform | T2 type, T2 value | Direct key access (trust type). Wrap arithmetic/parsing (value may fail). |
| `validated_dict["field"]` after boundary validation | T2 | Direct access. Already validated — trust it. |
| `request.json()`, `request.body()`, `request.form()` | T3 | Wrap in try/except. Validate structure. Return HTTP error on failure. |
| `response` from LLM/HTTP/API call | T3 | Wrap call. Validate response immediately. Coerce types at boundary. |
| `user_upload`, `user_yaml`, form field values | T3 | Validate at boundary. Reject on failure. |
| `oidc_token`, `jwt_claims` from external IdP | T3 | Verify signature. Validate claims. |
| `secret_key`, `passphrase`, `ServerSecretStore.*` | T1-eq | Never log values. Redact in errors. Crash on anomaly. |
| Pipeline YAML about to be passed to `load_settings_from_yaml_string()` | T1-eq | Must be fully validated first. Engine trusts what we hand it. |

### What am I writing?

| Pattern | Rule |
|---------|------|
| Building a `TransformResult.error(...)` | Include `reason`, relevant field name, and first ~200 chars of bad value. Set `retryable=False` for data problems, `True` for transient failures. |
| Catching an exception | Narrow to specific types. Never `except Exception`. Never swallow — record then re-raise or return error result. |
| Returning an HTTP error | Include enough detail for the caller to fix their request. Never include secret values. |
| Filling in a missing field from external data | **STOP.** Is the field absent? Use `None`. Do not infer from adjacent fields. Do not use a default that changes meaning. See fabrication test below. |
| Adding `.get(key, default)` | **STOP.** Why is the key missing? If it's our data (T1/T2), the key must exist — use `[key]`. If it's external data (T3), explain in a comment why the default is correct and meaning-preserving. |
| Adding `getattr(obj, "field", default)` | **STOP.** This is almost always wrong. If `obj` is a typed dataclass/model (T2), access `.field` directly. If you're unsure the field exists, that's a bug to fix, not a gap to paper over. |
| Adding `hasattr(obj, ...)` | **STOP.** Unconditionally banned. Use `isinstance()` for type dispatch or access the field directly. `hasattr` swallows all exceptions from `@property` getters. |
| Adding `try/except` around our own code | **STOP.** Is this operating on external data values, or is this our code calling our code? If the latter, let it crash — that's a bug. |
| Writing a default value for a missing external field | Run the fabrication test: (1) Would an auditor get a value the source never provided? (2) Could the source later provide a contradictory value? (3) Would `None` be more honest? If any answer is yes, use `None`. |

### Common Tier 2 value hazards (type-valid, operation-unsafe)

These are the "reality is greasy" cases. The type is correct but the operation
will blow up. Simple models often miss these because the type checks pass.

| Operation | Hazard | What to catch |
|-----------|--------|---------------|
| `a / b` | Division by zero | `ZeroDivisionError` |
| `int(s)`, `float(s)` | Unparseable string | `ValueError` |
| `datetime.fromisoformat(s)` | Invalid date format | `ValueError` |
| `json.loads(s)` on row field | Malformed JSON in a text field | `json.JSONDecodeError` |
| `s.encode("utf-8")` | Surrogate escapes from bad source data | `UnicodeEncodeError` |
| `d["key"]` on a JSON-parsed field | Parsed to list, not dict | `TypeError` (or validate structure) |
| `math.log(x)` | Zero or negative | `ValueError`, `math domain error` |
| `url.split("/")[3]` | Too few segments | `IndexError` |
| `re.match(pattern, s).group(1)` | No match returns None | `AttributeError` on `.group()` |
| `row["field"].strip()` | Field is None (nullable column) | `AttributeError` |

**All of these get wrapped in try/except with `TransformResult.error()`.** The row
is quarantined, not the pipeline. The pattern is always: catch the specific
exception, return an error result with the field name and value snippet, set
`retryable=False` (the data won't fix itself on retry).

### Forbidden patterns (never write these)

```python
# NEVER — defensive access on typed objects
getattr(config, "port", 8451)        # config.port exists or it's a bug
settings.get("secret_key", "")       # settings is a Pydantic model, not a dict
hasattr(row, "some_field")           # banned unconditionally

# NEVER — swallowing exceptions
except Exception:
    pass                              # evidence destroyed
except Exception:
    return default_value              # silent fabrication
except Exception as e:
    logger.warning(f"Failed: {e}")    # logged but lost — not in audit trail

# NEVER — fabricating absent data
value = external_response.get("field", 0)       # 0 is not absence
value = row.get("optional_field", "unknown")     # "unknown" is a lie
has_more = len(records) >= page_size             # inference, not observation

# NEVER — broad catch at Tier 1
try:
    data = json.loads(landscape_json)
except Exception:
    data = {}                         # EVIDENCE TAMPERING on our own data
```

## External Call Boundaries in Transforms

**CRITICAL:** Trust tiers are about **data flows**, not plugin types. **Any data crossing from an external system is Tier 3**, regardless of which plugin makes the call.

Transforms that make external calls (LLM APIs, HTTP requests, database queries) create **mini Tier 3 boundaries** within their implementation:

```python
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    # row enters as Tier 2 (pipeline data - trust the schema)

    # 1. External call creates Tier 3 boundary — wrap it
    try:
        llm_response = self._llm_client.query(prompt)  # EXTERNAL DATA - zero trust
    except Exception as e:
        return TransformResult.error({"reason": "llm_call_failed", "error": str(e)}, retryable=True)

    # 2. IMMEDIATELY validate at the boundary — retryable=False (bad data, not transient)
    try:
        parsed = json.loads(llm_response.content)
    except json.JSONDecodeError:
        return TransformResult.error({"reason": "invalid_json", "raw": llm_response.content[:200]}, retryable=False)

    if not isinstance(parsed, dict):
        return TransformResult.error({"reason": "invalid_json_type", "expected": "dict", "actual": type(parsed).__name__}, retryable=False)

    # 3. NOW it's our data (Tier 2) — validated, trust it
    output = {**row.to_dict(), "llm_classification": parsed["category"]}
    return TransformResult.success(output, success_reason={"action": "llm_classified"})
```

**The rule: Minimize the distance external data travels before you validate it.**

- Validate immediately — right after the external call returns
- Coerce once — normalize types at the boundary
- Trust thereafter — once validated, it's Tier 2 pipeline data
- Don't carry raw external data — passing `llm_response` to helper methods without validation
- Don't defer validation — "I'll check it later when I use it"
- Don't validate multiple times — if it's validated once, trust it

**Common external boundaries in transforms:**

| External Call Type | Tier 3 Boundary | Validation Pattern |
|-------------------|-----------------|-------------------|
| LLM API response | Response content | Wrap JSON parse, validate type is dict, check required fields |
| HTTP API response | Response body | Wrap request, validate status code, parse and validate schema |
| Database query results | Result rows | Validate row structure, handle missing fields, coerce types |
| File reads (in transform) | File contents | Same validation as source plugins |
| Message queue consume | Message payload | Parse format, validate schema, quarantine malformed messages |

## Coercion Rules by Plugin Type

| Plugin Type | Coercion Allowed? | Rationale |
|-------------|-------------------|-----------|
| **Source** | Yes | Normalizes external data at ingestion boundary |
| **Transform (on row)** | No | Receives validated data; wrong types = upstream bug |
| **Transform (on external call)** | Yes | External response is Tier 3 - validate/coerce immediately |
| **Sink** | No | Receives validated data; wrong types = upstream bug |

## Operation Wrapping Rules

| What You're Accessing | Wrap in try/except? | Why |
|----------------------|---------------------|-----|
| `self._config.field` | No | Our code, our config - crash on bug |
| `self._internal_state` | No | Our code - crash on bug |
| `landscape.get_row_state(token_id)` | No | Our data - crash on corruption |
| `checkpoint_data["tokens"]` | No | Our data - we wrote this JSON |
| `row["field"]` arithmetic/parsing | Yes | Their data values can fail operations |
| `external_api.call(row["id"])` | Yes | External system, anything can happen |
| `json.loads(external_response)` | Yes | External data - validate immediately |
| `validated_dict["field"]` | No | Already validated at boundary - trust it |

**Rule of thumb:**

- **Reading from Landscape tables?** Crash on any anomaly - it's our data.
- **Reading checkpoints or deserialized audit JSON?** Crash on any anomaly - it's our data.
- **Operating on row field values?** Wrap operations, return error result, quarantine row.
- **Calling external systems?** Wrap call AND validate response immediately at boundary.
- **Using already-validated external data?** Trust it - no defensive `.get()` needed.
- **Accessing internal state?** Let it crash - that's a bug to fix.

**Serialization does not change trust tier.** Data we wrote to our own database or checkpoint file is still Tier 1 when we read it back, even though it passes through `json.loads()` or SQLAlchemy deserialization. The trust boundary is about *who authored the data*, not the transport format. Checkpoints, audit records, and Landscape tables are all our data — we defined the schema, we wrote the values, we own the invariants. If a deserialized checkpoint is missing a `"tokens"` key or a `"row_id"` field, that is corruption in our system, not a data quality issue to handle gracefully. Crash immediately.

## Wrapping Discipline at Tier 3 Boundaries

Wrapping at a Tier 3 boundary involves **two independent decisions**. Conflating them is a common mistake:

| Decision | Question | Answer |
|----------|----------|--------|
| **1. Do I wrap?** | Is this an operation on external data? | Always yes at Tier 3 |
| **2. What do I catch?** | What can this operation actually throw? | Only the specific exceptions |

**Never resolve a "broad except" warning by removing the try/except.** The wrapping is correct — the catch clause is lazy. Fix the catch, not the wrap.

**Never resolve a "silent except" warning by swallowing to a default.** The catch should record what happened and re-raise, not silently degrade.

### The resolution ladder for CI warnings

| CI Rule | Warning | Wrong Fix | Right Fix |
|---------|---------|-----------|-----------|
| R4 (broad-except) | `except Exception` at Tier 3 boundary | Remove the try/except | Narrow to specific exceptions |
| R6 (silent-except) | Catch swallows without re-raise | Swallow to a default value | Record the error, then re-raise |

### Example: SDK serialization on Tier 3 response

```python
# WRONG — broad catch, silent swallow
try:
    raw_response = response.model_dump()
except Exception:
    raw_response = None  # Silently lost the error

# WRONG — removed the wrapping entirely
raw_response = response.model_dump()  # Crashes on malformed Tier 3 data

# RIGHT — specific catch, record-then-reraise
try:
    raw_response = response.model_dump()
except (TypeError, ValueError, RecursionError, AttributeError) as dump_exc:
    # Record minimal audit entry before re-raising
    recorder.record_call(
        state_id=state_id,
        call_index=call_index,
        call_type=CallType.LLM,
        status=CallStatus.ERROR,
        request_data=request_dto,
        error=LLMCallError(
            type="ResponseProcessingError",
            message=f"model_dump() failed: {dump_exc}",
            retryable=False,
        ),
        latency_ms=latency_ms,
    )
    raise LLMClientError(
        f"Failed to serialize LLM response: {dump_exc}",
        retryable=False,
    ) from dump_exc
```

**Why `model_dump()` needs wrapping:** The response object is Tier 3 — it came from an external provider. `model_dump()` operates on that external data. If the provider returned something malformed enough to break serialization, that's an external data problem, not a bug in our code. We wrap it, catch what it can throw, record what happened, and re-raise — we never crash silently, and we never pretend it didn't happen.

## Pipeline Templates as Tier 2 Data

Pipeline templates (Jinja2 prompt templates in YAML config) are **Tier 2 — user-provided, validated at load time, trusted during rendering**. This creates two distinct error categories:

| Failure Type | When | Effect | Example |
|-------------|------|--------|---------|
| **Structural** (parse) | Init time | Stop run — no row will ever succeed | `{% if unclosed` (syntax error) |
| **Operational** (render) | Per-row | Quarantine row, continue pipeline | `{{ row.missing_field }}` (undefined variable) |

**Rules:**

- Templates are parsed **once** at plugin construction (`__init__` / `__post_init__`), never re-parsed per-row
- `TemplateSyntaxError` during parse -> `TemplateError` propagates up -> run fails at setup
- `UndefinedError` / `SecurityError` during render -> `TemplateError` caught by transform -> `TransformResult.error()` -> row quarantined
- Per-query template overrides (multi-query LLM transforms) are pre-compiled in `MultiQueryStrategy.__post_init__`, not deferred to first row

**Why this matters:** A broken template can never produce valid results for any row. Deferring the parse error to render time would misclassify a config error (structural) as a data error (operational), quarantining the first row instead of stopping the run.

## Web Component Trust Model

The web server is an **operations interface**, not the audit backbone. Landscape
integrity is the engine's responsibility. The web component's job is to not make
things worse.

**Guiding principle:** The web layer is T2/T3 except for paths that handle secrets
or could compromise pipeline execution.

### Web Trust Tiers

| Trust Tier | What It Covers | Handling |
|-----------|---------------|----------|
| **Tier 1 (full trust)** | Nothing directly — the web component doesn't write to Landscape. It reads audit data via `query_repository` (read-only). | N/A |
| **Tier 2 (elevated trust)** | Server config values after validation, session state, authenticated user identity, pipeline YAML constructed by the composer. These passed our validators — if they're wrong, that's a bug in our code. | Expect correctness. Crash on type violations. |
| **Tier 3 (zero trust)** | User uploads, user-submitted pipeline YAML, HTTP request bodies, OIDC tokens from external IdPs, LLM responses in the composer, query parameters, form data. | Validate at boundary, reject on failure. |
| **Tier 1-equivalent** | Two specific paths: (1) Secret handling — `ServerSecretStore`, `secret_key`, passphrase resolution. A corrupted or leaked secret is catastrophic. (2) Anything that feeds into `load_settings_from_yaml_string()` for actual execution — once we hand config to the engine, it must be correct. | Crash immediately on any anomaly. No coercion, no fallbacks. |

### Web Tier Boundary Map

```text
BROWSER / CLIENT              WEB SERVER                    ENGINE
(Tier 3 — zero trust)         (Tier 2 — elevated trust)     (owns Tier 1)

┌─────────────────┐           ┌─────────────────┐           ┌─────────────────┐
│ HTTP requests    │           │ Validated config │           │ Landscape DB    │
│ User uploads     │──T3────►│ Session state    │──T1-eq──►│ Pipeline exec   │
│ Pipeline YAML    │ validate │ Auth identity    │ secrets & │ Audit trail     │
│ OIDC tokens      │ at       │ Composer output  │ validated │                 │
│ Form data        │ boundary │                  │ YAML only │                 │
└─────────────────┘           └─────────────────┘           └─────────────────┘
                                      │
                                      │ reads (read-only)
                                      ▼
                              ┌─────────────────┐
                              │ query_repository │
                              │ (Landscape read) │
                              └─────────────────┘
```

### Web-Specific Rules

| Scenario | Trust Tier | Correct Response |
|----------|-----------|------------------|
| Config file parse error | T2 (our format) | Hard failure — refuse to start |
| Env var has wrong type (e.g. `PORT=banana`) | T2 | Hard failure — Pydantic rejects |
| Unknown YAML key in config | T2 | Hard failure — typo detection |
| User uploads malformed CSV | T3 | Reject with error response |
| User submits invalid pipeline YAML | T3 | Validate, return errors, don't execute |
| OIDC token from external IdP | T3 | Verify signature, validate claims |
| Composer LLM response | T3 | Validate structure before using |
| `secret_key` value handling | T1-equivalent | Never log, never include in errors, redact in startup banner |
| Pipeline YAML passed to engine | T1-equivalent | Must be fully validated — engine trusts what we hand it |
| Config file might be hostile | Not a concern | If attacker has write access to config, they own the server |

### Difference from Pipeline Trust Model

The pipeline model's Tier 1 (Landscape) crashes on any anomaly because corrupted
audit data is evidence tampering. The web component has no Tier 1 of its own —
it's a consumer of Landscape (read-only) and a gateway to the engine. Its elevated
paths (T1-equivalent) exist only where the web layer could corrupt the engine's
Tier 1 data by handing it bad secrets or invalid pipeline config.

This means the web component can afford to be **more forgiving** than the engine
on operational issues (return HTTP errors instead of crashing) while being
**equally strict** on the two paths that bridge to the engine.

## Tier 3 Source Boundary: Coercion vs Fabrication

Sources are the ONLY place coercion is allowed. But coercion has limits.

### Coercion (meaning-preserving) — allowed

```python
# String to int — preserves the value
raw_value = "42"
coerced = int(raw_value)  # 42 — same meaning, different type

# String to bool — preserves the value
raw_value = "true"
coerced = raw_value.lower() in ("true", "1", "yes")  # True

# Whitespace normalization — preserves the value
raw_value = "  John Smith  "
coerced = raw_value.strip()  # "John Smith"
```

### Fabrication (meaning-changing) — forbidden

```python
# WRONG — None means "unknown", 0 means "zero". Different meanings.
raw_value = None
fabricated = raw_value or 0  # Downstream can't distinguish "zero" from "missing"

# WRONG — absence is evidence, not an invitation to infer
if "has_more_records" not in response:
    has_more = len(response.get("records", [])) >= page_size  # FABRICATION
    # The API said nothing about pagination. We guessed. The audit trail
    # now contains a confident answer to a question the source never answered.

# RIGHT — record the absence
has_more = response.get("has_more_records")  # None if absent — honest
```

### The fabrication decision test

Before filling in a missing field, ask:

1. **If an auditor queries this field, will they get a value the external system actually provided?** If no, it's fabrication.
2. **If the external system's behaviour changes and the field starts appearing with a different value than what we inferred, will the audit trail silently contain two contradictory sources of truth?** If yes, it's fabrication.
3. **Would recording `None` and letting the consumer handle absence be less convenient but more honest?** If yes, record `None`.

### Source quarantine pattern

```python
def process_row(self, raw_row: dict[str, Any], row_index: int) -> SourceResult:
    # Coerce where meaning is preserved
    try:
        amount = int(raw_row["amount"]) if raw_row.get("amount") is not None else None
    except (ValueError, TypeError):
        return SourceResult.quarantine(
            raw_row, reason=f"Row {row_index}: 'amount' not coercible to int: {raw_row['amount']!r}"
        )

    # Don't fabricate — None means "not provided"
    return SourceResult.success({
        "id": raw_row["id"],
        "amount": amount,  # Could be None — that's honest
    })
```

## Tier 2 Nuance: Type-Safe != Operation-Safe

```python
# Data is type-valid (int), but operation fails
row = {"divisor": 0}  # Passed source validation
result = 100 / row["divisor"]  # ZeroDivisionError - wrap this!

# Data is type-valid (str), but content is problematic
row = {"date": "not-a-date"}  # Passed as str
parsed = datetime.fromisoformat(row["date"])  # ValueError - wrap this!
```

### Correct wrapping for Tier 2 operations

```python
def process(self, row: PipelineRow, ctx: PluginContext) -> TransformResult:
    # Type is trustworthy (source validated it as int), but value might cause failure
    try:
        ratio = row["revenue"] / row["headcount"]
    except ZeroDivisionError:
        return TransformResult.error(
            {"reason": "division_by_zero", "field": "headcount", "row_id": row["id"]},
            retryable=False,  # Not transient — this row's data is the problem
        )

    try:
        event_date = datetime.fromisoformat(row["event_date"])
    except ValueError:
        return TransformResult.error(
            {"reason": "invalid_date", "field": "event_date", "value": row["event_date"][:50]},
            retryable=False,
        )

    return TransformResult.success({**row.to_dict(), "ratio": ratio, "event_date": event_date})
```

**Key distinction:** We don't coerce (that would hide the problem). We don't crash (that would kill the pipeline for one bad row). We quarantine and record what happened.

## Web Component Code Examples

### Tier 3: Validating user-submitted pipeline YAML

```python
async def submit_pipeline(request: Request) -> JSONResponse:
    # Request body is Tier 3 — user can send anything
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    yaml_content = body.get("yaml")
    if not isinstance(yaml_content, str):
        return JSONResponse(status_code=400, content={"error": "yaml must be a string"})

    # Validate before it reaches the engine (T1-equivalent gate)
    try:
        settings = load_settings_from_yaml_string(yaml_content)
    except (yaml.YAMLError, ValidationError) as exc:
        return JSONResponse(status_code=422, content={"error": str(exc)})

    # Only validated config reaches the engine
    await execution_service.start_run(settings)
```

### Tier 1-equivalent: Secret handling

```python
# WRONG — secrets in error messages
try:
    resolved = vault_client.get_secret(secret_name)
except VaultError as exc:
    raise HTTPException(500, f"Failed to resolve {secret_name}: key={api_key}")  # LEAKED

# RIGHT — redact everything, log the failure class only
try:
    resolved = vault_client.get_secret(secret_name)
except VaultError as exc:
    slog.error("secret_resolution_failed", secret_name=secret_name)  # Name only, not value
    raise HTTPException(500, "Secret resolution failed") from exc
```

### Tier 2: Trusting validated config

```python
# Config passed Pydantic validation — it's T2, trust it
config: ServerConfig = app.state.config

# WRONG — defensive access on our own validated config
port = getattr(config.server, "port", 8451)  # Why would this be missing?

# RIGHT — direct access, crash on bug
port = config.server.port  # If this fails, our model is broken — that's a bug
```
