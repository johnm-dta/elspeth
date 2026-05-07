# Freeze/Serialize Coherence — Design Spec

**Date:** 2026-03-21
**Status:** Draft
**Scope:** `contracts/hashing.py`, `contracts/results.py`, `contracts/plugin_context.py`, tests

## Problem

ELSPETH's contracts layer (L0) has two subsystems that are internally incoherent:

1. **`freeze.py`** — the standard immutability mechanism for frozen dataclasses. Converts `dict` → `MappingProxyType`, `list` → `tuple`, `set` → `frozenset`.
2. **`hashing.py`** — primitive-only canonical JSON and hashing. Passes data directly to `rfc8785.dumps()` with no normalization.

`rfc8785.dumps()` hard-rejects `MappingProxyType` with `CanonicalizationError`. This means frozen container types produced by `deep_freeze` cannot be hashed by their own layer's hashing module. Additionally, `_reject_non_finite()` checks `isinstance(obj, dict)` instead of `isinstance(obj, Mapping)`, so NaN inside frozen containers is silently missed.

The bug is latent — current call sites happen to thaw via `to_dict()` before hashing. But this relies on convention, and the Systems Thinking analysis identified it as a textbook "Shifting the Burden" archetype: each thaw call is symptomatic relief that removes pressure for the structural fix.

`core/canonical.py` (L1) accidentally works because it normalizes through the `Mapping` ABC before passing to rfc8785. This creates a second escape hatch that erodes the layer boundary.

### Related Bugs

- **ArtifactDescriptor shallow freeze** (`contracts/results.py:394`): Uses `MappingProxyType(dict(self.metadata))` instead of `deep_freeze()`. Nested mutable objects leak through.
- **`plugin_context.py:323-324` thaw-refreeze**: Thaws frozen data to create a defensive copy, then wraps in `RawCallPayload` which immediately re-freezes. Redundant cycle — `deep_freeze()` is idempotent.

## Design Decision

**Approach A: Make `contracts/hashing.py` handle frozen types natively.**

Rejected alternatives:

- **Approach B (thaw before serialize):** Codifies the Shifting the Burden pattern. Separates construction-and-freeze from hash-and-record, opening a conceptual gap in audit integrity. Convention enforcement is unreliable — Python's type system cannot express "dict but not MappingProxyType."
- **Hybrid:** No useful middle ground. Either the hashing module handles frozen containers or it doesn't.

### Rationale (from 4-agent specialist panel)

| Agent | Verdict | Key Argument |
|-------|---------|--------------|
| System Architect | **A** | L0 layer cohesion — if L0 defines frozen types, L0's hashing must handle them |
| Systems Thinker | **A** | Option B is Shifting the Burden archetype; thaw-before-hash separates construction from hashing |
| Python Engineer | B | Codebase already implements `to_dict()` pattern; enforce with type annotations |
| Quality Engineer | B | Easier to test; loud crash preferred over silent wrong hash |

The tie was broken by the project's stated goal of moving away from convention-based enforcement. The architect and systems thinker arguments align with ELSPETH's offensive programming philosophy: make the wrong thing impossible rather than document the right thing.

The Quality Engineer's concern about silent wrong hashes is addressed by the test strategy: property tests prove frozen and unfrozen produce identical hashes, and cross-module parity tests prevent the two normalizers from diverging.

## Scope

### Boundary: "Primitives and their frozen equivalents"

The expanded contract for `contracts/hashing.py`:

| Module | Handles |
|--------|---------|
| `contracts/hashing.py` | JSON primitives + stdlib frozen equivalents (`MappingProxyType` → `dict`, `tuple` already handled, `frozenset` → rejected with `TypeError`) |
| `core/canonical.py` | Everything above + domain types (pandas, numpy, `PipelineRow`) |

No pandas, numpy, or datetime handling enters `contracts/hashing.py`. That stays in `core/canonical.py`.

### `frozenset` Handling

Rejected with a clear `TypeError`. JSON has no set type, and canonical ordering of complex frozenset elements is non-trivial. No current frozen dataclass field holds a `frozenset` (audited — all usage is module-level constants and `ClassVar`). Deferred until a real use case exists.

### Important Distinction: Container Thawing vs DTO Serialization

This fix addresses **frozen container types** (`MappingProxyType`, `tuple`, `frozenset`) flowing into hashing functions. It does NOT eliminate `to_dict()` on frozen dataclass DTOs.

Frozen dataclasses like `LLMCallRequest`, `HTTPCallResponse`, etc. have `to_dict()` methods that perform two operations: (a) convert the dataclass instance to a dict structure, and (b) call `deep_thaw()` on frozen container fields. Operation (a) is structurally necessary — `rfc8785` cannot serialize dataclass instances, only dicts/lists/primitives. Operation (b) becomes unnecessary once `contracts/hashing.py` handles frozen containers.

However, these `to_dict()` methods serve multiple consumers (checkpoint serialization via `json.dumps`, telemetry exporters, etc.) that still require plain dicts. Therefore, `to_dict()` methods and their internal `deep_thaw()` calls remain unchanged. The fix ensures that if frozen container types *do* reach the hashing layer (now or in the future), they are handled correctly rather than crashing.

## Changes

### 1. `contracts/hashing.py` — Combined Normalize + Reject Traversal

Rename `_reject_non_finite()` to `_normalize_frozen_and_reject_non_finite()`. The new function performs a single recursive traversal that:

- Normalizes `MappingProxyType` → `dict` (recurse into values)
- Rejects `frozenset` with `TypeError` and clear message
- Rejects NaN/Infinity in floats with `ValueError`
- Passes through all other types unchanged
- Returns the normalized structure (not just validates)

The existing `_reject_non_finite()` already recurses through `dict` and `(list, tuple)`. The change widens `dict` to `Mapping` (the ABC from `collections.abc`) and adds `MappingProxyType` → `dict` conversion in the same pass. `tuple` is already handled at line 44: `isinstance(obj, (list, tuple))`.

The `canonical_json()` function calls the new traversal, which returns a normalized structure, then passes that to `rfc8785.dumps()`.

Update the module docstring from "primitive-only" to "primitives and their frozen equivalents."

### 2. `contracts/results.py` — ArtifactDescriptor Deep Freeze

Replace the shallow wrap in `__post_init__`:

```python
# Before (shallow — nested mutables leak)
object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

# After (deep — matches every other frozen dataclass)
object.__setattr__(self, "metadata", deep_freeze(self.metadata))
```

Add the `deep_freeze` import from `elspeth.contracts.freeze`.

### 3. `contracts/plugin_context.py:323-324` — Eliminate Thaw-Refreeze Cycle

Current code thaws frozen data to create a defensive copy, then wraps in `RawCallPayload` which immediately `deep_freeze()`s it again. Since `deep_freeze()` is idempotent on already-frozen data, pass the frozen data directly to `RawCallPayload`.

Two sub-changes required:

**3a.** Remove the `deep_thaw()` calls at lines 323-324. Pass `request_data` and `response_data` directly to `RawCallPayload()` at lines 340-341.

**3b.** Widen the `isinstance` check at line 334 from `isinstance(raw_usage, dict)` to `isinstance(raw_usage, Mapping)`. Without the thaw, `raw_usage` extracted via `.get("usage")` may be a `MappingProxyType`. The `Mapping` ABC (from `collections.abc`) catches both `dict` and `MappingProxyType`. The `.get("usage")` call at line 333 already works on `MappingProxyType` since it supports the `Mapping` protocol.

Import `Mapping` from `collections.abc` at the top of the function or module.

### 4. Surviving `to_dict()` / `deep_thaw()` — NOT Changed

These remain for legitimate reasons:

#### 4a. DTO `to_dict()` Methods (Structural Necessity)

Frozen dataclass DTOs (`LLMCallRequest`, `HTTPCallResponse`, `CallData`, etc.) have `to_dict()` methods that convert dataclass instances to plain dicts. This is DTO serialization, not container thawing — `rfc8785` cannot serialize dataclass instances. These methods also serve multiple consumers beyond hashing:

| Consumer | Why plain dict required |
|----------|----------------------|
| `checkpoint_dumps()` → `json.dumps()` | stdlib JSON rejects `MappingProxyType` |
| Telemetry exporters (Datadog, OTLP, console) | `isinstance(value, dict)` checks, `json.dumps()` |
| DeepDiff comparison (`verifier.py:365`) | Compares container types — frozen vs plain creates false diffs |

Specific surviving call sites:

| Site | File | Reason |
|------|------|--------|
| Call payload hashing | `execution_repository.py:533-549` | DTO-to-dict serialization for `stable_hash` + `canonical_json` + payload store |
| Error/context serialization | `execution_repository.py:279-283` | `ExecutionError.to_dict()` (frozen dataclass), `NodeStateContext.to_dict()` (Protocol) |
| LLM telemetry hashing (3 sites) | `llm.py:294,438,518` | DTO-to-dict for `stable_hash` in telemetry events |
| HTTP telemetry hashing (4 sites) | `http.py:316,351,532,606` | DTO-to-dict for `stable_hash` in telemetry events |
| Checkpoint serialization (4 sites) | `checkpoint/manager.py`, `aggregation.py`, `coalesce_executor.py` | `checkpoint_dumps()` → `json.dumps()` |
| Telemetry event serialization | `events.py:411-413` | Exporters do `isinstance(value, dict)` checks |
| Schema field export | `exporter.py:241` | Exporter yields `Iterator[dict]` — callers use `json.dumps()` |
| DeepDiff comparison | `verifier.py:365` | Third-party library compares container types |

#### 4b. Why This Is Acceptable

The `to_dict()` pattern is not the bug this spec fixes. The bug is that `contracts/hashing.py` cannot handle frozen container types at all. Once the hashing module handles `MappingProxyType`, the system has a correct structural foundation. The `to_dict()` methods remain because they serve a broader purpose (DTO serialization for multiple consumers), not because of a hashing limitation.

Future work could reduce `to_dict()` usage by:
- Adding a frozen-type-aware JSON encoder to `checkpoint_dumps()` (eliminates checkpoint thaw)
- Updating telemetry exporters to use `Mapping` ABC instead of `isinstance(dict)` checks
- These are independent improvements, not part of this fix.

## Test Strategy

### T1. Hash Equivalence Property Test

**Location:** `tests/property/canonical/test_freeze_hash_equivalence.py` (new file)

Core invariant: for all JSON-like structures `x`:
```
contracts.hashing.canonical_json(deep_freeze(x)) == contracts.hashing.canonical_json(x)
contracts.hashing.stable_hash(deep_freeze(x)) == contracts.hashing.stable_hash(x)
```

Uses the existing `json_values` Hypothesis strategy from `tests/strategies/json.py`. This proves frozen and unfrozen data produce identical hashes — the fundamental guarantee that makes Approach A safe.

### T2. Frozen-Type Unit Tests

**Location:** New class `TestFrozenTypeHandling` in `tests/unit/contracts/test_hashing.py`

- `MappingProxyType({"a": 1})` succeeds, output equals `canonical_json({"a": 1})`
- Nested: `MappingProxyType({"a": MappingProxyType({"b": 2})})` succeeds
- Mixed: `MappingProxyType({"items": (1, 2, 3)})` succeeds
- `frozenset({1, 2})` raises `TypeError` with message about frozenset
- NaN inside `MappingProxyType({"x": float("nan")})` raises `ValueError`
- Infinity inside `MappingProxyType({"x": float("inf")})` raises `ValueError`

### T3. Cross-Module Parity Test

**Location:** Extend existing `TestCanonicalJsonConsistency` in `tests/unit/contracts/test_hashing.py`

For frozen inputs, both modules must agree:
```
contracts.hashing.canonical_json(frozen_x) == core.canonical.canonical_json(frozen_x)
```

This prevents the two normalizers from diverging. Test with simple, nested, and deeply nested frozen structures.

### T4. Round-Trip Contract Tests

**Location:** Per-dataclass test files (extend existing test classes)

For each frozen dataclass with a `to_dict()` method, verify that hashing the thawed output equals hashing the frozen equivalent:
```
contracts.hashing.canonical_json(instance.to_dict()) == contracts.hashing.canonical_json(deep_freeze(instance.to_dict()))
```

This proves both paths (thaw-then-hash via `to_dict()`, and direct frozen hash) produce identical results.

### T5. Plugin Context Thaw-Refreeze Elimination

**Location:** `tests/unit/plugins/test_context.py` (extend existing)

- Verify `record_call()` works when `request_data`/`response_data` contain `MappingProxyType` values (frozen containers passed directly without thaw)
- Verify token usage extraction works when `raw_usage` is `MappingProxyType` (the `isinstance(raw_usage, Mapping)` change)
- Verify `RawCallPayload` receives and freezes the data correctly without intermediate thaw

## Fail-Forward Commitment

All items in this spec ship together or none of them do. No partial states. If any part of the implementation proves harder than expected, we push through it — the design is sound and the complexity is implementation detail.

## Relationship to Other Work

- **Batch 4 P1: SchemaContract KeyError migration** — separate ticket, not in scope
- **Batch 4 P1: SourceRow wrong exception type** — separate ticket, not in scope
- **Batch 3: `_reject_non_finite` gap in hashing.py** — resolved by this fix (same root cause)
- **Batch 4 P1: RoutingAction.reason → canonical_json** — resolved by this fix (hashing now handles MappingProxyType)
- **Batch 4 P1: ArtifactDescriptor shallow freeze** — resolved by this fix (change 2)
- **Code quality sweep epic** (`elspeth-6bf1d1179d`) — this fix addresses the systemic pattern identified across batches 3 and 4
