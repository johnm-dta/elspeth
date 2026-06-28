# T17: Split PluginContext into Phase-Based Protocols

**Date:** 2026-02-26
**Status:** Approved
**Issue:** elspeth-rapid-c42eca
**Epic:** RC3.3 Architectural Remediation

## Problem

`PluginContext` is a 20-field god-object passed to every plugin (sources, transforms, sinks). Every plugin receives the full surface area regardless of what it needs:

- Simple transforms (passthrough, field_mapper, truncate) receive 20 fields but access **zero** of them
- Sources need `record_validation_error()` but receive `state_id`, `token`, checkpoint API, etc.
- Sinks need `contract` and `landscape` but receive `rate_limit_registry`, `batch_token_ids`, etc.

This creates implicit coupling and makes it impossible to know what a plugin actually depends on without reading its implementation.

## D2 Revision: Phase-Based Split

The original D2 decision proposed field-category grouping (IdentityContext / AuditContext / ExecutionContext). Three independent code analyses mapped every field access across all 42 plugin files and found this doesn't match actual usage — most complex plugins need fields from all three categories.

The revised split groups by **consumer role** (what each plugin phase actually needs):

### Protocol Definitions

```python
@runtime_checkable
class SourceContext(Protocol):
    """What sources need: identity + validation recording."""
    run_id: str
    node_id: str | None
    operation_id: str | None
    landscape: LandscapeRecorder | None
    telemetry_emit: Callable[[Any], None]

    def record_validation_error(...) -> ValidationErrorToken: ...
    def record_call(...) -> Call | None: ...

@runtime_checkable
class TransformContext(Protocol):
    """What transforms need: per-row identity + call recording + checkpoint."""
    run_id: str
    state_id: str | None
    node_id: str | None
    token: TokenInfo | None
    batch_token_ids: list[str] | None
    contract: SchemaContract | None

    def record_call(...) -> Call | None: ...
    def get_checkpoint() -> BatchCheckpointState | None: ...
    def set_checkpoint(state: BatchCheckpointState) -> None: ...
    def clear_checkpoint() -> None: ...

@runtime_checkable
class SinkContext(Protocol):
    """What sinks need: contract resolution + call recording."""
    run_id: str
    contract: SchemaContract | None
    landscape: LandscapeRecorder | None
    operation_id: str | None

    def record_call(...) -> Call | None: ...

@runtime_checkable
class LifecycleContext(Protocol):
    """What on_start()/on_complete() need: infrastructure references."""
    run_id: str
    node_id: str | None  # [R1] Added per review — set before on_start(), avoids future cascade
    landscape: LandscapeRecorder | None
    rate_limit_registry: RateLimitRegistry | None
    telemetry_emit: Callable[[Any], None]
    payload_store: PayloadStore | None
    concurrency_config: RuntimeConcurrencyConfig | None
```

### Why Phase-Based

| Approach | Pros | Cons |
|----------|------|------|
| Field-category (Identity/Audit/Execution) | Clean conceptual separation | Most plugins need all 3; doesn't actually narrow |
| Phase-based (Source/Transform/Sink/Lifecycle) | Matches actual access patterns; real narrowing | Less conceptually pure |
| Status quo (single PluginContext) | No work | No enforcement, god-object persists |

Evidence: field access inventory across all 42 plugin files shows phase-based boundaries align with actual usage while field-category boundaries do not.

## Existing State

### Dead Fields (to remove first)

| Field | Evidence |
|-------|----------|
| `plugin_name` | Declared, never set anywhere in codebase |
| `llm_client` | TYPE_CHECKING import only, never assigned |
| `http_client` | TYPE_CHECKING import only, never assigned |
| `tracer` | Phase 3 placeholder; telemetry implemented via SpanFactory instead |
| `start_span()` | Only caller was tracer; zero production usage |
| `get()` | Zero production callers |

### Construction Sites (3 total)

| Location | Purpose |
|----------|---------|
| `engine/orchestrator/core.py:1205` | Main run construction |
| `engine/orchestrator/core.py:2180` | Resume run construction |
| `engine/orchestrator/export.py:92` | Export sink context |

### Consumption: ~100+ Method Signatures

- **Sources:** 4 plugins, ~14 methods
- **Simple transforms:** 5 plugins, ~5 methods (none access ctx)
- **Complex transforms:** 3 plugins (web_scrape, prompt_shield, content_safety), ~10 methods
- **LLM transforms:** 3 plugins (transform, openrouter_batch, azure_batch), ~22 methods
- **Sinks:** 4 plugins, ~20 methods
- **Engine:** executors (4), orchestrator (4 modules), processor, operations — ~20 methods

### Test Impact

- 1,279 PluginContext references across 99 test files
- 444 direct `PluginContext(...)` constructions in 69 files
- `make_context()` factory exists but only used by 4 files
- 3 `_TestablePluginContext` subclasses in integration tests
- 1 `FakePluginContext` in property tests (already protocol-friendly)

## Design Constraints

1. **Concrete `PluginContext` remains.** Executors mutate it in-place between steps (`ctx.state_id = ...`, `ctx.token = ...`). Protocols narrow the read surface; the implementation stays mutable.
2. **`record_call()` stays on the concrete class.** Its 180-line implementation accesses fields from multiple protocols. Each protocol exposes it as a method signature; the implementation lives on PluginContext.
3. **`on_start()` is wider than `process()`.** Complex transforms capture infrastructure in `on_start(ctx: LifecycleContext)` and store references privately. `process(row, ctx: TransformContext)` receives the narrower per-row context.
4. **Executors keep accepting concrete `PluginContext`.** They mutate fields, so they need the full mutable class, not a protocol.
5. **Tests mostly don't change.** Direct `PluginContext(...)` constructions still work because the concrete class satisfies all protocols. Only test method signatures that accept protocol types would narrow.

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| `record_call()` crosses protocol boundaries | High | Keep implementation on concrete class; protocols declare method signature only |
| Executor mutation requires concrete type | High | Executors explicitly typed as `PluginContext`, not protocol |
| `on_start()` vs `process()` type mismatch | Medium | Separate `LifecycleContext` protocol for lifecycle hooks |
| 444 test constructions | Low | Concrete class unchanged; tests still work. Only signature narrowing is optional |
| Checkpoint API only used by azure_batch | Low | Include in TransformContext; no-op default for non-batch transforms |
| `plugin_name="test"` in contract test bases breaks after Phase 0 | High | [B1] Add contract test base fixtures to Phase 0 cleanup list |
| Defensive fallback removal breaks tests that skip `on_start()` | Medium | [W1] Test audit step before removing fallbacks in Phase 2-3 |
| `on_start()` → `process()` ordering now load-bearing | Medium | [N3] Strictly better (fail-fast), document behavioral change |
| `payload_store` not wired in orchestrator constructions | Low | [R6] Pre-existing; T17 surfaces it earlier via `on_start()` |

## Engine-Internal Methods Not in Protocols

> **[R4] Review finding:** `record_transform_error()` exists on `PluginContext` and is called by the engine
> (`engine/executors/transform.py:461`, `engine/processor.py:1001,1049`), but appears in zero protocols.
> This is correct — engine calls it, not plugins. Documented here to prevent future confusion.
> The disjointness tests do not cover it. If it is ever needed by plugins, add to `TransformContext`.

## Protocol Implementation Notes

> **[N4]** `record_call()` implementation on `PluginContext` internally accesses `self.telemetry_emit`.
> Because both live on the concrete class, this works despite `telemetry_emit` not being in every protocol.
> `SinkContext` correctly omits `telemetry_emit` from its surface — no sink code accesses `ctx.telemetry_emit` directly. **[N5]**
>
> **[N7]** `isinstance()` with `@runtime_checkable` Protocol only checks that attribute *names* exist at runtime,
> not their signatures or types. mypy's structural check IS the enforcing mechanism. The `isinstance` checks
> in alignment tests are a weaker guarantee — they confirm name presence, mypy confirms full structural conformance.
> Both checks are valuable but serve different purposes.

## Subtasks

| # | Task | Effort | Risk | Phase |
|---|------|--------|------|-------|
| 1 | Remove 5 dead fields + 2 unused methods | S | Low | 0: Cleanup |
| 2 | Define 4 protocol interfaces in `contracts/contexts.py` | M | Med | 1: Protocols |
| 3 | Add protocol alignment + disjointness tests | M | Low | 1: Protocols |
| 4 | Update `plugins/protocols.py` signatures | M | Med | 2: Signatures |
| 5 | Update `plugins/base.py` signatures | S | Med | 2: Signatures |
| 6 | Update source plugins (4 files) | M | Low | 3: Plugins |
| 7 | Update simple transform plugins (5 files) | S | Low | 3: Plugins |
| 8 | Update complex transforms (web_scrape, prompt_shield, content_safety) | M | Med | 3: Plugins |
| 9 | Update LLM transforms (transform, openrouter_batch, azure_batch) | L | High | 3: Plugins |
| 10 | Update sink plugins (4 files) | M | Med | 3: Plugins |
| 11 | Update batching infrastructure (mixin.py) | M | Med | 4: Engine |
| 12 | Update executors (transform, sink, gate, aggregation) | M | High | 4: Engine |
| 13 | Update orchestrator + operations.py | M | High | 4: Engine |
| 14 | Update test factories + bulk test signatures | L | Low | 5: Tests |
| 15 | Full verification: mypy + ruff + contracts + test suite | M | Low | 6: Gate |

## Verification Strategy

1. **mypy strict mode** — protocols verified structurally at compile time (primary enforcement mechanism **[N7]**)
2. **Protocol alignment tests** — modeled on `test_config_alignment.py` precedent, with mechanical introspection **[R3]** and real minimal-object discrimination tests **[R2]**
3. **Bidirectional field coverage** — every PluginContext field in at least one protocol or in explicit `EXECUTOR_ONLY_FIELDS` allowlist **[R3]**
4. **Plugin contract tests** — existing `TransformContractTestBase` extended for protocol verification
5. **Full regression** — all ~10,000 tests must pass (with **[B1]** contract test base fixture fix in Phase 0)
6. **Config contracts checker** — `.venv/bin/python -m scripts.check_contracts`
7. **Test lifecycle audit** — verify `on_start()` is called before `process()` in complex transform tests before removing defensive fallbacks **[W1]**
