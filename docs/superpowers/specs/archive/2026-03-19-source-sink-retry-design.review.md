# Architecture Review: Source and Sink I/O Retry Design

**Date:** 2026-03-19
**Reviewer:** Architecture Critic Agent
**Spec:** `docs/superpowers/specs/2026-03-19-source-sink-retry-design.md`
**Verdict:** Good design with a few issues that need resolution before implementation.

---

## Confidence Assessment

| Area | Confidence | Basis |
|------|-----------|-------|
| Layer dependency analysis | High | Verified against actual protocol/engine code |
| Audit trail recording | High | Cross-referenced `SourceContext`, `SinkContext`, `record_call` signatures |
| Trust model compliance | High | Verified against CLAUDE.md three-tier model |
| Config pipeline compliance | High | Verified against existing `RuntimeRetryConfig` and protocol patterns |
| File inventory completeness | Medium | Checked key files; some plugin files not fully inspected |

## Risk Assessment

| Risk | Severity | Likelihood |
|------|----------|------------|
| `getattr` usage violates project rules | Medium | Certain (spec line 211) |
| `RuntimeIORetryConfig` missing `jitter` field | Medium | Certain (spec lines 277-293) |
| Key collision on composite keys with `\|` separator | Low | Unlikely but possible |
| Wrapper `on_start`/`on_complete`/`close` delegation gap | Medium | Certain (spec omits) |

---

## Findings

### 1. `getattr` in Default Predicate Violates Project Rules -- Medium

**Evidence:** Spec lines 210-211:
```python
retryable = getattr(exc, "retryable", None)
```

**Problem:** CLAUDE.md unconditionally bans `hasattr` and forbids defensive `getattr` with defaults. The spec's justification (lines 222-223: "interface polymorphism, not defensive programming") attempts to reframe the usage, but the actual code is `getattr(exc, "retryable", None)` -- a textbook defensive pattern that silently returns `None` for any exception type that doesn't carry the attribute. This is exactly what the tier model enforcer flags.

**The spec's own argument undermines itself:** It says `retryable` is "a documented protocol on system-owned exception classes." If it's a protocol, define it as a Protocol and use `isinstance` to check conformance. If third-party exceptions don't carry the flag, that's what the type-based fallback on lines 215-220 handles.

**Recommendation:** Replace `getattr` with `isinstance` check against a `RetryableError` protocol or base class:

```python
if isinstance(exc, RetryableError):  # Our protocol/base
    return exc.retryable
# Fall through to type-based defaults
```

This is also more explicit -- an auditor reading the predicate sees "check if it's one of our classified exceptions, then check known transient types" instead of "probe any exception for an arbitrary attribute."

### 2. `RuntimeIORetryConfig` Missing `jitter` Field -- Medium

**Evidence:** Spec lines 277-293 define `RuntimeIORetryConfig` with 5 fields: `max_attempts`, `base_delay`, `max_delay`, `exponential_base`, `enabled`. The existing `RuntimeRetryConfig` (at `contracts/config/runtime.py:130-156`) has 5 fields: `max_attempts`, `base_delay`, `max_delay`, `exponential_base`, `jitter`.

**Problem:** The spec says (line 296) "This satisfies `RuntimeRetryProtocol` -- reuses `RetryManager` internally." But `RuntimeRetryProtocol` requires a `jitter` property (verified at `contracts/config/protocols.py:68`). `RuntimeIORetryConfig` as specified does not have `jitter`, so it does NOT satisfy `RuntimeRetryProtocol`. The wrappers cannot pass `RuntimeIORetryConfig` to `RetryManager` as-is.

**Additionally:** The spec adds an `enabled` field that `RuntimeRetryProtocol` does not define. If the intent is to reuse `RuntimeRetryProtocol`, `enabled` is extraneous (wrapping is conditional at the orchestrator, not inside `RetryManager`). If the intent is a new `RuntimeIORetryProtocol`, the spec needs to document that `jitter` is also required (as an internal default, matching the existing pattern).

**Recommendation:** Either:
- Add `jitter: float` to `RuntimeIORetryConfig` with the same internal default (1.0), or
- Use `RuntimeRetryConfig` directly (it already has everything needed) and keep `enabled` as a separate field on `PipelineConfig`, or
- Define `RuntimeIORetryProtocol` explicitly and include `jitter`.

### 3. Layer Dependency: L2 Composing L3 -- Acceptable

**Evidence:** Spec line 72: "Both wrappers live at L2 (engine) in a new `engine/retryable_io.py`. They compose L3 plugins but don't depend on any specific plugin."

**Assessment:** This is acceptable. The wrappers depend on `SourceProtocol` and `SinkProtocol` (L0 contracts), not on any L3 implementation. The orchestrator (L2) already holds `SourceProtocol` and `SinkProtocol` references in `PipelineConfig` (`engine/orchestrator/types.py:89-91`). Wrapping them at L2 is structurally identical to the existing pattern. No layer violation.

The spec correctly identifies that allowlist entries are needed (`config/cicd/enforce_tier_model/`, line 420) -- the `getattr` usage will trigger the tier model enforcer. If finding 1 is resolved (replacing `getattr` with `isinstance`), this allowlist entry may not be needed.

### 4. Audit Trail: `record_call` Without `state_id` Context -- Low

**Evidence:** Spec lines 101, 144: Retry decisions recorded via `ctx.record_call()`.

**Assessment:** `SourceContext.record_call()` exists and doesn't require a `state_id` (verified at `contracts/contexts.py:73-83`). Source-level calls are attributed to the source `node_id` and `operation_id`, not to a per-row state. This is correct -- the retry decision is a source-level event, not a row-level event.

`SinkContext.record_call()` similarly exists (`contracts/contexts.py:157-165`). The sink write context carries `operation_id` for attribution.

**Audit primacy is satisfied:** The spec records retry decisions before executing retries (lines 101, 144). The Landscape gets the record synchronously. Inner source/sink calls produce their own audit records. No audit gaps.

### 5. Wrapper Lifecycle Delegation Not Specified -- Medium

**Evidence:** The spec covers `load()` and `write()` delegation but does not address how `SourceWrap` and `SinkWrap` delegate `on_start()`, `on_complete()`, `close()`, `flush()`, or protocol attributes like `name`, `output_schema`.

**Problem:** `SourceProtocol` requires `name`, `output_schema`, `on_start()`, `on_complete()`, `close()`, `load()`. `SinkProtocol` requires `name`, `output_schema`, `on_start()`, `on_complete()`, `close()`, `write()`, `flush()`, plus `header_mode`, `idempotent`. The wrappers must delegate all of these to remain transparent to the orchestrator.

The spec mentions (line 128) that the wrapper captures `LifecycleContext` from `on_start()`, but doesn't specify that `on_start()` itself is delegated through. The orchestrator calls `source.on_start(ctx)` during run setup (`core.py:1500`). If `SourceWrap` doesn't delegate `on_start`, the inner source never initializes.

**Recommendation:** Add a section stating that all protocol methods and attributes are delegated to the inner plugin, with the only behavioral override being `load()` (for `SourceWrap`) and `write()` (for `SinkWrap`). Specify that `on_start()` is intercepted to capture the `LifecycleContext` AND delegated to the inner source.

### 6. Composite Key Separator Collision -- Low

**Evidence:** Spec line 120: `"|".join(str(row[f]) for f in fields)`

**Problem:** The spec claims `|` is "separator-safe for GUIDs" (line 120). This is true for GUIDs specifically. But the feature is generic -- any source can declare `source_key_fields`. If a future source uses string fields as composite keys where values contain `|`, keys will collide: `("a|b", "c")` and `("a", "b|c")` both produce `"a|b|c"`.

**Practical risk is low** because the current use case is Dataverse GUIDs. But the spec presents this as a general mechanism.

**Recommendation:** Either (a) use a collision-resistant separator like `\x00` (null byte, impossible in most string data), or (b) use tuple hashing (`hash(tuple(str(row[f]) for f in fields))`) with a set of ints, or (c) document the constraint ("key fields must not contain the `|` character").

### 7. Error Classification Consistent with Trust Model -- Correct

**Evidence:** Spec lines 237-243: Framework bugs, audit integrity errors, and programming errors are never retried. Lines 207-221: Transient I/O errors are retried.

**Assessment:** This correctly separates:
- Tier 1 violations (`AuditIntegrityError`) -- crash immediately, never retry
- System bugs (`FrameworkBugError`, `KeyError`, etc.) -- crash immediately
- Tier 3 boundary failures (network, auth, rate limiting) -- retry with backoff

The classification is consistent with CLAUDE.md's trust model. The `classify_retryable` override (lines 227-235) allows plugins to classify domain-specific errors (e.g., Dataverse-specific HTTP status codes) without the engine needing to know about them.

### 8. Config Pipeline Follows Settings->Runtime Pattern -- Partially

**Evidence:** Spec lines 249-304 define `IORetrySettings` (Pydantic) and `RuntimeIORetryConfig` (frozen dataclass) with `from_settings()`.

**Assessment:** The two-layer pattern is correctly followed. `from_settings()` provides explicit field mapping with rename (`initial_delay_seconds` -> `base_delay`, `max_delay_seconds` -> `max_delay`). The spec includes alignment documentation (`contracts/config/alignment.py`, line 392) and alignment tests (line 422).

**Gap:** The `jitter` issue from finding 2 must be resolved for this to actually work with `RetryManager`.

### 9. File Inventory Gaps -- Medium

**Evidence:** Spec lines 377-434.

**Missing items:**

1. **`plugins/sources/dataverse.py` config model:** The spec adds `source_key_field` as a FetchXML config option (line 363) but doesn't list changes to the Dataverse source's Pydantic config model (the `options` schema). This is implementation detail but belongs in the file inventory since it's a schema change.

2. **`contracts/plugin_protocols.py` default values:** The spec says `supports_retry` defaults to `True` and `source_key_fields` defaults to `None` (lines 170-174). Protocol classes in Python don't carry default values. The defaults live in `BaseSource`/`BaseSink` (`plugins/infrastructure/base.py`). The spec lists `base.py` changes (line 399) but conflates protocol definition with default implementation.

3. **No mention of `core/config.py` integration point:** `IORetrySettings` needs to be added as a field on the top-level settings model (presumably `PipelineSettings` or similar in `core/config.py`). Line 388 says "Add `IORetrySettings`" but doesn't specify where in the settings hierarchy it's nested.

### 10. Memory Growth on Full Re-run Sources -- Low

**Evidence:** Spec line 108: "If `key_fields` is `None`: yield all rows (full re-run)."

**Problem:** For sources without key fields (CSV, JSON), a retry re-yields ALL rows including those already processed. The orchestrator has already assigned `token_id`s and created `node_states` entries for the first-pass rows. The spec says (line 322) "Row index continuity: The orchestrator's `row_index` continues monotonically" -- but on full re-run without key tracking, the wrapper yields rows that were already processed. The orchestrator would create duplicate tokens for the same source rows.

**Wait -- this may not be a real problem** for CSV/JSON sources because they read from static files. A transient failure during CSV read is rare (disk I/O error?). But the spec doesn't address what happens if a non-keyed source IS retried. Are duplicate rows acceptable? The spec doesn't say.

**Recommendation:** Clarify the duplicate-row semantics for non-keyed source retry. Either (a) document that full re-run produces duplicates and the orchestrator's `row_id` assignment handles dedup, or (b) add a positional skip for static sources (which the spec already notes is safe for static files, line 35).

---

## Strengths

- **Content-based key skip is the right design.** The positional skip rejection (lines 26-35) is well-reasoned and correctly identifies the eventual consistency problem with OData pagination. This is a genuinely hard problem and the spec handles it well.
- **Two-level retry separation is clean.** Client-level 401 retry (immediate, single attempt) vs. wrapper-level transient retry (backoff, multi-attempt) is a correct decomposition that avoids retry amplification.
- **Idempotency guard on sink retry is essential** and correctly implemented. Non-idempotent sinks refuse retry rather than risking duplicate writes.
- **No new dependencies.** Reusing `RetryManager` and tenacity avoids introducing a parallel retry mechanism.
- **Security section** (lines 465-469) correctly identifies that entity IDs stay in-memory only and credential material is never stored in retry state.

---

## Information Gaps

1. The spec does not address interaction with the existing checkpoint/resume mechanism during a retry. If a checkpoint fires mid-retry, does the checkpoint capture the wrapper's `processed_keys` state? The spec says keys are NOT checkpointed (line 123), but doesn't address what happens if a checkpoint fires DURING a retry cycle.
2. The spec does not address what happens if `inner.on_start()` fails during retry reconstruction (line 104). Is that a retryable failure or a fatal error?
3. The spec does not address telemetry emission for retry events. CLAUDE.md requires that infrastructure lifecycle events (including retry) be telemetered. The spec records to audit trail but doesn't mention `telemetry_emit`.

## Caveats

- This review is based on the spec text and cross-referenced against the actual codebase. Implementation may reveal additional issues not visible from the spec alone.
- The `RetryManager` internals were not fully inspected -- the claim that `RuntimeIORetryConfig` can be passed to it depends on protocol satisfaction that has a known gap (finding 2).
- Plugin-specific `classify_retryable` implementations were not assessed since they don't exist yet.
