# `_compose_loop` decomposition plan

**Target**: `src/elspeth/web/composer/service.py::ComposerServiceImpl._compose_loop`, lines 1931–3546 (1,616 lines).
**Mode**: Refactor (in-place; method becomes a thin driver that calls four extracted phase helpers; public surface — `compose()` — unchanged).
**Branch**: `compose-loop-decomp` worktree off RC5.2.
**Scope**: planning only. The writer agent executes this plan in a follow-up session.

> The hypothesis in the brief (`call_model → dispatch_tools → update_state → flush_audit`) does not survive contact with the code. The actual data flow has **five** phases, not four, and `update_state` is not a separable phase — `state` is reassigned *inside* the tool-dispatch loop body and is at its final value the moment that for-loop exits. Section 1 below documents the real seams; the rest of the plan is written against them.

---

## 1. Loop anatomy

### 1.1 Method entry → outer iteration

| Concern | Lines | Notes |
|---|---|---|
| Signature + docstring | 1931–1960 | 10 params, returns `ComposerResult` |
| One-time prelude (skill upsert, message build, recorder, actor) | 1961–1992 | Runs before the `while True:` |
| Pre-loop counter/cache decls | 1994–2044 | See §1.3 |
| `while True:` outer | 2046 | Single open-ended loop; terminations are `return` or `raise` |
| Method end | 3546 | Closing `continue` of the final `else:` of the budget-classification block (line 3540) |

### 1.2 Per-iteration control-flow skeleton

Line numbers below are absolute file lines (not offsets).

```
2046  while True:
2050      _emit_progress(model_call)
2051      response = await _call_llm_before_deadline(...)        # LLM call #1
2059      assistant_message = response.choices[0].message
2062      if len(assistant_tool_calls) > _max_tool_calls_per_turn:
2063          self._telemetry.tool_call_cap_exceeded_total.add(1) # only telemetry counter in body
2064          raise ComposerConvergenceError.capture(...)
2079      if not assistant_message.tool_calls:                   # NO-TOOLS branch
2080–2149     four repair-injection branches, each `repair_turns_used += 1; continue`
2151          _emit_progress(complete)
2161          result = await _finalize_no_tool_response(...)
2180          return replace(result, repair_turns_used=..., persisted_assistant_message_id=..., persisted_tool_call_turn=...)

2187      _emit_progress(tool_batch)
2195      llm_messages.append({"role": "assistant", ...})        # appended ONCE per turn before dispatch
2217      turn_has_mutation = False; turn_has_discovery = False
2219      all_cache_hits = True
2226      tool_outcomes: list[_ToolOutcome] = []
2227      plugin_crash = None; plugin_crash_cause = None
2229      pre_state_id = current_state_id
2231      decoded_args_by_call_id = {}
2232–2238 turn_sessions_service / turn_session_uuid / turn_preferences (one-shot per turn)
2239      proposals_this_turn = 0

2241      for tool_call in assistant_message.tool_calls:
              # Pre-dispatch validation (three ARG_ERROR sites):
2266–2311     - JSON decode failure        → recorder.record(finish_arg_error) + continue
2313–2362     - Non-dict arguments         → recorder.record(finish_arg_error) + continue
2372–2410     - Canonicalisation failure   → recorder.record(finish_arg_error) + continue

2413–2465     # Discovery cache hit         → recorder.record(finish_success, cache_hit=True) + continue
2477–2512     # Schema-required-paths      → recorder.record(finish_arg_error) + continue

2514–2628     # Explicit-approve proposal interception (proposals_this_turn++, turn_has_mutation, continue)

2646–2921     # Advisor escape-hatch (request_advisor_hint)
                 - disabled / budget-exhausted   → recorder.record + continue
                 - arg-error / deadline timeout  → recorder.record + continue (some RAISE ComposerConvergenceError)
                 - success                       → recorder.record(finish_success) + continue

2938–2959     # Session-aware tool (request_interpretation_review) — helper rebinds state, last_validation

2966–2994     # preview_pipeline preflight (may RAISE ComposerRuntimePreflightError.capture)

3051–3093     # Closure binding for dispatch (default args capture loop locals)
3093–3100     outcome = await dispatch_with_audit(...)           # SYNC tool execution under audit envelope
3101–3147     except ToolArgumentError                            → recorder already wrote; continue
3148–3190     except (AssertionError, MemoryError, RecursionError, SystemError) → RAISE (Tier-1 invariant)
3191–3194     except AuditIntegrityError                          → RAISE
3195–3248     except Exception                                    → plugin_crash = capture(); plugin_crash_cause = exc; BREAK

3256–3313     # Success path:  state = result.updated_state; last_validation = ...; last_runtime_preflight = ...
              # mutation_success_seen, anti_anchor.record_success/failure, _serialize_tool_result,
              # llm_messages.append({"role": "tool", ...}), turn_has_mutation/discovery
3314      self._phase3_last_tool_outcomes = tuple(tool_outcomes)      # test hook

              # ↑↑↑ End of inner for-loop ↑↑↑

3315–3380 # Redaction (pure async work; builds redacted_assistant_tool_calls + redacted_tool_rows)
3380–3381 self._phase3_last_redacted_assistant_tool_calls = ...; self._phase3_last_redacted_tool_rows = ... # test hooks

3382–3420 # DB persistence:
              audit_outcome = await sessions_service.persist_compose_turn_async(...)
              except AuditIntegrityError                          → set failed_turn + RAISE
              current_state_id = audit_outcome.current_state_id
              failed_turn = FailedTurnMetadata(...)
              two AuditIntegrityError raises if invariants broken
              persisted_assistant_message_id = audit_outcome.assistant_id
              persisted_tool_call_turn = True

3421–3436 # Plugin-crash propagation (RAISE post-persist; uses persisted_tool_call_turn discipline)

3446–3461 # Anti-anchor hint injection (mutates llm_messages, emits progress)

3465–3466 if all_cache_hits: continue                              # NO budget charge

3475–3527 if turn_has_mutation:
              composition_turns_used += 1
              if composition_turns_used >= _max_composition_turns:
                  # B-4D-3 LAST-CHANCE LLM call (LLM call #2 within same iteration)
                  response = await _call_llm_before_deadline(...)
                  assistant_message = response.choices[0].message
                  if not assistant_message.tool_calls:
                      result = await _finalize_no_tool_response(...)
                      return replace(result, ...)
                  RAISE ComposerConvergenceError.capture(budget=composition)
3528–3539 elif turn_has_discovery:
              discovery_turns_used += 1
              if discovery_turns_used >= _max_discovery_turns:
                  RAISE ComposerConvergenceError.capture(budget=discovery)
3540–3545 else:
              # advisor-only turn — no counter charged; continue
              continue
```

### 1.3 Mutated loop-local inventory (15 variables, not 12)

Counted across the entire `_compose_loop` body (lines 1931–3546). Carriers are grouped by phase consumer.

| # | Variable | Declared | Type | Writers | Readers (key sites) | Carried across iterations? |
|---|---|---|---|---|---|---|
| 1 | `composition_turns_used` | 1994 | `int` | 3476 (`+=1`) | 2065, 3477, 3520, 3532 | **Yes** |
| 2 | `discovery_turns_used` | 1995 | `int` | 3529 (`+=1`) | 2065, 3520, 3530, 3532 | **Yes** |
| 3 | `mutation_success_seen` | 1996 | `bool` | 3283 (success path) | 2170, 3510 | **Yes** |
| 4 | `last_validation` | 2007 | `ValidationSummary \| None` | 2958 (session-aware), 3259 (success) | 2599, 3055 (closure capture) | **Yes** |
| 5 | `last_runtime_preflight` | 2014 | `ValidationResult \| None` | 3260 (success) | 2166, 3461, 3506 | **Yes** |
| 6 | `advisor_calls_used` | 2033 | `int` | 2780 (`+=1`) | 2679, 2685, 2802, 2826, 2894 | **Yes** |
| 7 | `repair_turns_used` | 2040 | `int` | 2094, 2112, 2148, 2121 (`+=1` × 4) | 2081, 2097, 2119, 2142, 2181, 3517 | **Yes** |
| 8 | `persisted_assistant_message_id` | 2041 | `str \| None` | 3419 | 2183, 3516 | **Yes** |
| 9 | `persisted_tool_call_turn` | 2042 | `bool` | 3420 | 2184, 3422, 3500, 3517, 3524, 3536 | **Yes** |
| 10 | `failed_turn` | 2043 | `FailedTurnMetadata \| None` | 3360 (per-tool, inside dispatch), 3397 (persist), 3405 (persist) | 3429, 3526, 3538 | **Yes** |
| 11 | `current_state_id` | 2044 | `str \| None` | 3404 (persist) | 2229, 2583, 2947, 3392 | **Yes** |
| 12 | `state` | param | `CompositionState` (immutable, but rebound) | 2912 (session-aware), 3258 (success) | every dispatch site + persist + classify | **Yes** |
| 13 | `llm_messages` | 1971 | `list[dict]` (mutable list) | 2195 + 25+ `.append` sites | LLM call site (2051), repair injections, anti-anchor hint | **Yes** (accumulating chat history) |
| 14 | `discovery_cache` | 2001 | `dict[str, _CachedDiscoveryPayload]` | 3299 | 2415 (hit), 2429 | **Yes** |
| 15 | `runtime_preflight_cache` | 2013 | (cache object) | mutated inside `_cached_runtime_preflight` | 2967 | **Yes** |
| 16 | `recorder` | 1976 | `BufferingRecorder` | 30+ `.record()` sites | LLM-call recording + every error path's `recorder.invocations` / `recorder.llm_calls` | **Yes** (per-call) |
| 17 | `anti_anchor` | 2022 | `AntiAnchorTracker` | `record_failure` / `record_success` / `consume_fire` ≈ 18 sites | 3446 (should_fire) | **Yes** |

**Per-iteration scratch** (rebound at the top of each `while True:` body — these are the carriers between phases *within* one iteration):

| # | Variable | Declared (line in body) | Lifespan |
|---|---|---|---|
| A | `response` | 2051, also 3481 | call_model → dispatch (used in `_dispatch_session_aware_tool`); also re-bound in B-4D-3 |
| B | `assistant_message` | 2059, also 3489 | call_model → dispatch → persist (`.content`, `.tool_calls`) |
| C | `raw_assistant_content` | 2060 | dispatch → persist (3388) |
| D | `assistant_tool_calls` | 2061 | dispatch (3399 audit) |
| E | `turn_has_mutation` | 2217 | dispatch → classify (3475) |
| F | `turn_has_discovery` | 2218 | dispatch → classify (3528) |
| G | `all_cache_hits` | 2219 | dispatch → classify (3465) |
| H | `tool_outcomes` | 2226 | dispatch → persist (3326) |
| I | `plugin_crash` | 2227 | dispatch → persist (3394) → classify-skipping raise (3421) |
| J | `plugin_crash_cause` | 2228 | dispatch → persist (3431) |
| K | `pre_state_id` | 2229 | dispatch only (also written to `self._phase3_last_expected_current_state_id` at 2230 — a test hook) |
| L | `decoded_args_by_call_id` | 2231 | dispatch → persist (3328) |
| M | `turn_sessions_service`, `turn_session_uuid`, `turn_preferences` | 2232–2238 | dispatch only (explicit-approve branch) |
| N | `proposals_this_turn` | 2239 | dispatch only (2530 cap check) |

This is the real "12+" the prior reviewer alluded to: 14 per-iteration scratch carriers between phases, plus the 11 multi-iteration counters/buffers above the loop.

### 1.4 Phase boundaries (proposed, evidence-grounded)

The four-name hypothesis collapses two distinct phases (dispatch vs. update_state) and silently absorbs the *no-tool* termination path. The actual seams the data flow exposes:

| Phase | Lines | Output of phase | Why a seam |
|---|---|---|---|
| **P1. `call_model_turn`** | 2050–2076 | `assistant_message`, `assistant_tool_calls`, `raw_assistant_content` (+ may raise convergence on cap) | Single `_call_llm_before_deadline` + cap check. Pure consumer of `llm_messages`. No state mutations. |
| **P2. `try_terminate_no_tools`** | 2079–2185 | `ComposerResult \| None` (None = continue; non-None = early return) | The four repair-injection branches all mutate `llm_messages` + `repair_turns_used` and `continue`. The final non-repair branch returns. This must run *before* P3 because it short-circuits the rest of the iteration. |
| **P3. `dispatch_tool_batch`** | 2187–3313 | `_DispatchOutcome` carrier (see §2) — `tool_outcomes`, final `state`, final `last_validation`, final `last_runtime_preflight`, `plugin_crash`, `plugin_crash_cause`, `turn_has_mutation`, `turn_has_discovery`, `all_cache_hits`, `decoded_args_by_call_id`, `assistant_message`, `raw_assistant_content` | The `_ToolOutcome` Step-1-vs-Step-2 split at line 2220–2226 ("Step 1 — execute tool calls in async land while accumulating immutable _ToolOutcome records. Step 2 performs audit writes from this list; cancellation before Step 2 leaves the DB unchanged") is the prior author's own seam. `state` mutation IS in this phase — the only thing P4 needs is the final value. |
| **P4. `persist_turn_audit`** | 3314–3436 | `_PersistOutcome` carrier — new `current_state_id`, `persisted_assistant_message_id`, `persisted_tool_call_turn`, `failed_turn`. **Raises if `plugin_crash` was set.** | Redaction (pure functions) + `persist_compose_turn_async` + plugin_crash re-raise discipline. Conceptually one boundary because: redaction depends on tool_outcomes shape; persist depends on redacted output; plugin_crash propagation depends on persist's `persisted_tool_call_turn`. |
| **P5. `classify_and_budget_turn`** | 3438–3545 | `_ClassifyOutcome` carrier — continues outer loop OR returns a `ComposerResult` OR raises convergence | Anti-anchor fire + cache-hit short-circuit + counter bumps + B-4D-3 last-chance LLM call. B-4D-3 itself calls `_call_llm_before_deadline` a second time and may invoke `_finalize_no_tool_response` — extracting it cleanly is what makes the budget logic finally readable. |

**Divergence from the brief's four-name hypothesis** (call this out explicitly to the writer):

1. The brief's `update_state` does not exist as a separate phase — `state` mutates inside `dispatch_tool_batch` (line 3258, line 2912). The carrier from P3→P4 carries `state` as a field; P4 does not mutate it; P5 reads it for convergence-raise envelopes. No extracted "update_state" function would have a job.
2. The brief omits the no-tools termination path entirely. Phase P2 is load-bearing — without it, `dispatch_tool_batch` has to internally branch on "are there tool_calls?", which mixes termination logic with dispatch logic.
3. The brief's `flush_audit` conflates *in-memory* recorder discipline (BufferingRecorder, populated continuously across all five phases) with *DB-write* discipline (`persist_compose_turn_async`, fired once at end of P3 boundary). They are different invariants. P4 only owns the DB write; recorder is a long-lived per-call object that every phase reads from and writes to, and stays in the outer driver.

---

## 2. Handoff dataclass design

**Module location**: new file `src/elspeth/web/composer/_compose_loop_carriers.py` (L3, sibling of `service.py`). Importing the carriers into `service.py` introduces no new layer dependencies.

All carriers must follow the project's `freeze_fields` contract (CLAUDE.md "Frozen Dataclass Immutability"). `CompositionState` is itself a frozen-immutable type — do not double-freeze it. `_ToolOutcome` (in `web/sessions/_persist_payload.py`) is already frozen with its own `freeze_fields` guard — a `tuple[_ToolOutcome, ...]` field needs no further work because `tuple` is hashable and its members are deep-frozen by their own contracts.

### 2.1 `_CallModelOutcome` (P1 → P3)

```python
from dataclasses import dataclass
from typing import Any
from litellm.types.utils import Message  # litellm response choice type

@dataclass(frozen=True, slots=True)
class _CallModelOutcome:
    """Result of one LLM call in the compose loop.

    P2 (try_terminate_no_tools) reads ``has_tool_calls`` and the full
    ``assistant_message`` to decide whether to short-circuit. P3
    (dispatch_tool_batch) reads ``assistant_message.tool_calls`` and
    ``raw_assistant_content``. P5's B-4D-3 last-chance path produces
    a second instance per iteration.
    """

    assistant_message: Any           # litellm Message; opaque to ELSPETH
    raw_assistant_content: str | None
    assistant_tool_calls: tuple[Any, ...]   # litellm ToolCall objects, never mutated
    has_tool_calls: bool

    # NO freeze_fields required: `assistant_message` and tool_call objects
    # are litellm-owned (Tier 3 boundary value) and we treat them as opaque;
    # `assistant_tool_calls` is already a tuple. `raw_assistant_content` is
    # str|None. `has_tool_calls` is bool. All scalar/opaque — frozen=True
    # alone is sufficient.
```

### 2.2 `_TerminateOutcome` (P2 → driver)

```python
@dataclass(frozen=True, slots=True)
class _TerminateOutcome:
    """Decision from try_terminate_no_tools.

    - ``action == "continue"``  : driver re-enters P1 next iteration
    - ``action == "return"``    : driver returns ``result``
    """

    action: str  # Literal["continue", "return"]
    result: Any | None = None  # ComposerResult when action == "return"

    # No container fields — frozen=True only.
```

### 2.3 `_DispatchOutcome` (P3 → P4 → P5)

This is the **load-bearing carrier**. It replaces the 14 per-iteration scratch locals (§1.3 table B-N) with a single visible boundary.

```python
from collections.abc import Mapping
from elspeth.contracts.freeze import freeze_fields
from elspeth.core.composer.state import CompositionState
from elspeth.core.composer.validation import ValidationSummary
from elspeth.core.composer.runtime_preflight import ValidationResult
from elspeth.web.composer.errors import ComposerPluginCrashError
from elspeth.web.sessions._persist_payload import _ToolOutcome

@dataclass(frozen=True, slots=True)
class _DispatchOutcome:
    """Result of dispatching one tool batch.

    Carries everything P4 (persist) and P5 (classify) need to read. State
    mutation happens inside P3 (per tool call); ``state`` here is the
    *final* value after every successful tool call's ``updated_state``
    rebind. P4 does not mutate state; P5 reads it only for convergence-
    raise envelopes.
    """

    # State at end of dispatch
    state: CompositionState
    last_validation: ValidationSummary | None
    last_runtime_preflight: ValidationResult | None

    # Tool outcomes (already deep-frozen by _ToolOutcome.__post_init__)
    tool_outcomes: tuple[_ToolOutcome, ...]

    # Per-tool decoded arguments, by tool_call.id (used by P4 redaction)
    decoded_args_by_call_id: Mapping[str, Mapping[str, Any]]

    # Turn classification (set during dispatch, read by P5)
    turn_has_mutation: bool
    turn_has_discovery: bool
    all_cache_hits: bool

    # Plugin-crash carrier — P4 must propagate this AFTER persist
    plugin_crash: ComposerPluginCrashError | None
    plugin_crash_cause: BaseException | None

    # LLM call result threaded through — P4 reads `.content` and `.tool_calls`
    # for redaction/persist; P5 unused.
    assistant_message: Any
    raw_assistant_content: str | None
    assistant_tool_calls: tuple[Any, ...]

    # mutation_success_seen rebinds in dispatch's success path; carry the
    # delta so the driver can update its multi-iteration accumulator.
    mutation_success_observed: bool

    def __post_init__(self) -> None:
        # `decoded_args_by_call_id` is a dict[str, dict] at construction;
        # freeze_fields(deep_freeze) walks both the outer dict and each
        # nested arguments dict into MappingProxyType. tool_outcomes is
        # already a tuple of frozen _ToolOutcome; passing it to
        # freeze_fields is identity-preserving (deep_freeze short-circuits
        # on already-frozen containers per `contracts/freeze.py`).
        freeze_fields(self, "decoded_args_by_call_id", "tool_outcomes")
```

**Why `decoded_args_by_call_id` needs explicit freezing**: at construction in the loop body, line 2365 stores raw `dict[str, Any]` mappings keyed by tool_call.id. CLAUDE.md's "frozen dataclass with `dict` field MUST freeze in `__post_init__`" rule applies. `_ToolOutcome.call.function.arguments` is a string (the raw LLM JSON), not the decoded dict — these are distinct and both need to survive to P4.

### 2.4 `_PersistOutcome` (P4 → P5)

```python
from elspeth.web.composer.failed_turn import FailedTurnMetadata

@dataclass(frozen=True, slots=True)
class _PersistOutcome:
    """Result of redact-and-persist.

    Raises ComposerPluginCrashError before constructing this carrier when
    plugin_crash was set in DispatchOutcome AND persistence succeeded — so
    the carrier never carries a "post-crash" state; if construction happens
    at all, persist completed and no crash was pending.
    """

    current_state_id: str | None
    persisted_assistant_message_id: str | None
    persisted_tool_call_turn: bool
    failed_turn: FailedTurnMetadata | None

    # No container fields — frozen=True only.
```

### 2.5 `_ClassifyOutcome` (P5 → driver)

```python
@dataclass(frozen=True, slots=True)
class _ClassifyOutcome:
    """Decision from classify_and_budget_turn.

    - ``action == "continue"``     : driver re-enters P1
    - ``action == "return"``       : driver returns ``result``
    - (raises) ComposerConvergenceError on exhausted budget — does not
      reach this carrier
    """

    action: str  # Literal["continue", "return"]
    result: Any | None = None
    # Counters and flags updated within this phase need to flow back to
    # the driver so the multi-iteration state stays in one place:
    composition_turns_delta: int = 0  # 0 or 1
    discovery_turns_delta: int = 0    # 0 or 1
```

**Note on counters**: deltas, not absolute values. Keeps the driver as the owner of the multi-iteration counters; P5 is stateless across iterations.

### 2.6 Carriers that are NOT extracted

These stay as parameters/closures on the helper functions, because they are either:

- **Multi-iteration accumulators owned by the driver** (`composition_turns_used`, `discovery_turns_used`, `mutation_success_seen`, `advisor_calls_used`, `repair_turns_used`, `persisted_assistant_message_id`, `persisted_tool_call_turn`, `failed_turn`, `current_state_id`): these are not "carried between phases" — they are the driver's bookkeeping. Phase helpers receive them as parameters and return updates as deltas (P5) or replacement values (P4's `current_state_id`, `persisted_assistant_message_id`).
- **Long-lived service objects** (`recorder`, `anti_anchor`, `discovery_cache`, `runtime_preflight_cache`, `llm_messages`, `tools`): created once in the driver prelude and threaded by reference into every phase helper. `recorder` and `anti_anchor` are *write targets* for every phase; that is by design and is not a refactoring smell — see §5.

---

## 3. Behaviour preservation test inventory

### 3.1 Tests that pin specific branches of `_compose_loop`

| Test file | Branch pinned |
|---|---|
| `tests/unit/web/composer/test_compose_loop_audit_wiring.py` | P3 audit envelope (SUCCESS / ARG_ERROR / PLUGIN_CRASH sequences); B2 narrow-class rethrow records before raise; canonical_json fallback; preview preflight failure records tool invocation; CancelledError records plugin crash; non-finite-args ARG_ERROR; canonicalization sentinel diagnostic |
| `tests/unit/web/composer/test_compose_loop_anti_anchor.py` | P5 anti-anchor fire (3-identical-failures); discovery-success doesn't break anchor; mutation-success breaks anchor; 2 failures don't fire |
| `tests/unit/web/composer/test_compose_loop_llm_audit.py` | P1 LLM-call audit metadata (success / cost / reasoning / malformed / seed-omission / deadline timeout / bad request / unclassified exception / empty choices); P3 second LLM-call after tool result |
| `tests/unit/web/composer/test_compose_loop_persistence.py` | P3 Step-1 (3 tools succeed; arg-error continues; assertion error reraises; plugin bug captures crash); P4 Step-2 redaction via manifest walker, summarizer, raw-content absent, existing current_state_id, one persist call per turn, no legacy add_message |
| `tests/unit/web/composer/test_compose_loop_tool_call_cap.py` | P1 cap-exceeded raises + counter increments |
| `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py` | P3 session-aware dispatch (10 cases including pending-event placeholder, missing-state arg-error, F5a startup assert, F5c skill markdown idempotent, F6 rate-cap branch emits telemetry+writes audit row) |
| `tests/unit/web/composer/test_compose_loop_unknown_tool_name.py` | P4 redaction raises for unknown tool name |
| `tests/unit/web/composer/test_compose_loop_envelope.py` | (file empty of `test_` defs — likely fixtures only; writer to confirm) |
| `tests/unit/web/composer/test_compose_loop_test_driver.py` | run_one_turn_for_test requires wired sessions_service |
| `tests/unit/web/composer/test_advisor_tool.py` | P3 advisor-hint branch (disabled / budget exhausted / arg-error / timeout / success / advisor error payload) |
| `tests/unit/web/composer/test_audit_failure_primacy.py` | AuditIntegrityError propagation discipline through compose |
| `tests/unit/web/composer/test_compose_service_structure.py` | Whole-method structural assertions (writer must read before any move) |
| `tests/unit/web/composer/test_service.py` | Service-level behaviour, includes `compose()` happy paths |

### 3.2 Property and integration tests

| Test file | Property |
|---|---|
| `tests/property/web/composer/test_compose_loop_invariants.py` | **Hypothesis state machine** — must be the safety net between every move. Includes `test_compose_loop_audit_machine_examples`, failure-injection strategy contains required arms, all arms mechanically drivable, OTel counter postconditions |
| `tests/integration/web/test_compose_loop_concurrent_sessions.py` | DIFFERENT sessions don't deadlock; SAME session serialises via advisory lock; advisory lock acquired on Postgres; cross-allocator save-state serialises with persist-turn; concurrent persist-compose-turn same-state stale rejects without integrity alert |
| `tests/integration/web/test_compose_loop_latency_sanity.py` | per-turn p95 < 250ms with 8 tool calls — **regression risk for the redact-then-persist boundary** |
| `tests/integration/web/test_inv_audit_ahead_backward.py` | Audit ordering invariants |
| `tests/integration/web/test_blobs_ready_hash_postgres.py` | Blob-store mutation tool path through compose |
| `tests/integration/web/composer/guided/test_progressive_disclosure.py` | Guided→freeform handoff (`guided_terminal` parameter) |
| `tests/integration/pipeline/test_composer_llm_eval_characterization.py` | End-to-end eval characterisation against live (or mocked) LLM |
| `tests/unit/evals/test_convergence_scenarios_mocked_llm.py` | Convergence scenarios |

### 3.3 Gaps to flag for the writer

1. **B-4D-3 last-chance branch** (lines 3477–3527): the writer must search test names for `last_chance`, `B-4D-3`, `budget_exhausted_composition`, and `composition_turns >= max_composition_turns` patterns. The hypothesis-state-machine property tests likely cover *that* a convergence is raised, but may not cover the "second LLM call succeeds with no tool_calls → return `_finalize_no_tool_response`" path. If absent, **add a characterization test before extracting P5**.

2. **Advisor-timeout-exceeds-compose-deadline** (lines 2811–2818): the convergence-from-advisor-timeout path is the only place in P3 where the dispatcher itself raises `ComposerConvergenceError` (not `ComposerPluginCrashError` or `ComposerServiceError`). The `test_advisor_tool.py` tests cover the success / timeout / error payloads but not necessarily the case where the advisor's `advisor_deadline_limited=True` branch fires the compose-deadline raise. Writer to confirm via a `grep -n "advisor_deadline_limited\|COMPOSE_TIMEOUT" tests/unit/web/composer/test_advisor_tool.py`.

3. **The four no-tools repair branches** (lines 2080–2149): all four have at least the *positive* test path. The cross-product of "first repair branch fires, second one matches, third doesn't" is unlikely to be tested. Acceptable risk; do not add more tests for this.

4. **`self._phase3_last_*` test-hook attributes** (lines 2230, 3314, 3380, 3381, 3403): these are written inside the loop and read by test fixtures. Search `grep -rn "_phase3_last_" tests/` before moving to confirm which tests depend on them and at what call points; the carriers should not break these read sites.

---

## 4. Sequenced move plan

Every step must pass the **full safety net**:

```bash
.venv/bin/python -m pytest tests/property/web/composer/test_compose_loop_invariants.py \
    tests/unit/web/composer/test_compose_loop_audit_wiring.py \
    tests/unit/web/composer/test_compose_loop_anti_anchor.py \
    tests/unit/web/composer/test_compose_loop_llm_audit.py \
    tests/unit/web/composer/test_compose_loop_persistence.py \
    tests/unit/web/composer/test_compose_loop_tool_call_cap.py \
    tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py \
    tests/unit/web/composer/test_compose_loop_unknown_tool_name.py \
    tests/unit/web/composer/test_compose_loop_test_driver.py \
    tests/unit/web/composer/test_advisor_tool.py \
    tests/unit/web/composer/test_audit_failure_primacy.py \
    tests/unit/web/composer/test_compose_service_structure.py \
    tests/unit/web/composer/test_service.py \
    -x
```

Integration tests (`tests/integration/web/test_compose_loop_*`) run **at the end of each step** when reachable locally; minimum requirement is property + unit-compose between every step.

### Step 0 — Pre-flight characterization tests
*Goal: close test-coverage gaps identified in §3.3 before any structural change.*

- **Adds**: characterization tests for the B-4D-3 last-chance branch (return path AND raise path) if not already present.
- **Removes**: nothing.
- **Gate**: new tests pass against current `_compose_loop` (commit before any extraction).

### Step 1 — Introduce carriers module (no extraction yet)
- **Adds**: `src/elspeth/web/composer/_compose_loop_carriers.py` containing the five `@dataclass(frozen=True, slots=True)` carriers (§2.1–2.5), with `freeze_fields` guards.
- **Removes**: nothing.
- **Gate**: `mypy src/` clean; full safety net green (no behaviour change yet — the carriers are dead code).
- **Diff size**: ≈100 lines added, all in the new file.

### Step 2 — Extract `_call_model_turn` (P1)
- **Adds**: `async def _call_model_turn(self, llm_messages, tools, state, initial_version, deadline, recorder, progress, message) -> _CallModelOutcome` as a method on `ComposerServiceImpl`. The body is lines 2050–2076 with `raise` replaced by `raise` (cap-exceeded branch stays).
- **Removes**: those lines from `_compose_loop`; replaces with `outcome = await self._call_model_turn(...)` then unpacks fields.
- **Gate**: safety net green.
- **Diff size**: ≈40 lines moved + ≈10 lines of unpack.

### Step 3 — Extract `_try_terminate_no_tools` (P2)
- **Adds**: `async def _try_terminate_no_tools(self, *, message, llm_messages, state, session_id, repair_turns_used, current_state_id, user_id, last_runtime_preflight, runtime_preflight_cache, session_scope, mutation_success_seen, recorder, progress, initial_version, persisted_assistant_message_id, persisted_tool_call_turn, assistant_message) -> _TerminateOutcome | None`.
- **Removes**: lines 2079–2185 from `_compose_loop`. Replaces with `term = await self._try_terminate_no_tools(...)`; if `term is not None and term.action == "return"`: return `term.result`; if `term is not None and term.action == "continue"`: `repair_turns_used += 1; continue`.
- **Note**: this helper must return the `repair_turns_used` delta because all four repair branches `+= 1`. Carry it as a third field of `_TerminateOutcome` if `action == "continue"`. *(Update the carrier spec accordingly — see §2.2 note: add `repair_turns_delta: int = 0`.)*
- **Gate**: safety net green; specifically `test_compose_loop_interpretation_review_dispatch.py::test_fresh_session_set_pipeline_then_request_interpretation_review_persists_pending_event` and the proof-repair tests.
- **Diff size**: ≈110 lines moved.

### Step 4 — Extract `_persist_turn_audit` (P4) BEFORE dispatch
*Rationale: P4 is the smallest and most self-contained phase (no closures, no per-iteration scratch beyond `tool_outcomes` and `decoded_args_by_call_id`). Extracting it first reduces the surface area of P3's extraction in Step 5.*

- **Adds**: `async def _persist_turn_audit(self, *, dispatch: _DispatchOutcome, session_id, current_state_id, initial_version, recorder, persisted_tool_call_turn) -> _PersistOutcome`. Body is lines 3315–3420. The plugin-crash propagation (lines 3421–3436) stays in the driver — `_PersistOutcome` carries fields, the driver decides whether to raise based on `dispatch.plugin_crash is not None`.
- **Removes**: those lines.
- **Gate**: safety net green; specifically `test_compose_loop_persistence.py` (all Step-2 cases) and `test_audit_failure_primacy.py`.
- **Diff size**: ≈110 lines moved.

### Step 5 — Extract `_dispatch_tool_batch` (P3)
*The largest and most coupling-intensive phase. Do it after P1/P2/P4 because that leaves the driver small enough to make this extraction's diff readable.*

- **Adds**: `async def _dispatch_tool_batch(self, *, call_model: _CallModelOutcome, state, last_validation, last_runtime_preflight, llm_messages, recorder, anti_anchor, discovery_cache, runtime_preflight_cache, session_id, user_id, current_state_id, user_message_id, actor, initial_version, deadline, progress, session_scope, advisor_calls_used) -> tuple[_DispatchOutcome, int]`. The second tuple element is the new `advisor_calls_used` (since the dispatcher writes it and the driver owns it across iterations).
- **Removes**: lines 2187–3313 (the entire dispatch block including the assistant-message append at 2195, all the pre-dispatch one-shot bindings, the for-loop over tool_calls, every branch through advisor / session-aware / preview / general dispatch, and the success-path state mutation at 3258).
- **Gate**: safety net green; specifically every audit-wiring case, every advisor case, every interpretation-review dispatch case, every persistence Step-1 case.
- **Diff size**: ≈1,100 lines moved — by far the largest. The writer should consider splitting it into Step 5a / 5b if any test fails mid-extraction; suggested sub-split is "pre-dispatch validation arms" (2266–2512) vs. "tool-specific arms + success path" (2514–3313).
- **Risk**: highest of any step. The §7.7 anti-anchor `record_success` / `record_failure` ordering relative to `recorder.record(finish_*)` and `await _emit_progress(...)` must be preserved exactly. The writer must diff per-arm before/after.

### Step 6 — Extract `_classify_and_budget_turn` (P5)
- **Adds**: `async def _classify_and_budget_turn(self, *, dispatch: _DispatchOutcome, persist: _PersistOutcome, llm_messages, tools, state, recorder, anti_anchor, progress, message, initial_version, deadline, last_runtime_preflight, runtime_preflight_cache, session_scope, user_id, mutation_success_seen, composition_turns_used, discovery_turns_used, max_composition_turns, max_discovery_turns) -> _ClassifyOutcome`.
- **Removes**: lines 3438–3545.
- **Gate**: safety net green; specifically anti-anchor tests, B-4D-3 last-chance characterization test from Step 0, and the convergence tests.
- **Diff size**: ≈100 lines moved.

### Step 7 — Driver simplification + docstring update
- **Removes**: the now-redundant `# Step 1 — ... Step 2 — ...` inline comments and the multi-paragraph rationale comments that documented the carriers' invariants (those move to the dataclass docstrings in `_compose_loop_carriers.py`).
- **Adds**: a one-paragraph driver docstring naming the five phases and pointing at the carrier module.
- **Gate**: safety net green; **also run** `tests/integration/web/test_compose_loop_concurrent_sessions.py`, `tests/integration/web/test_compose_loop_latency_sanity.py`, `tests/integration/web/test_inv_audit_ahead_backward.py`.
- **Diff size**: ≈100 lines removed, ≈30 added.

### Step 8 — Final verification
- `.venv/bin/python -m pytest tests/` (full suite).
- `.venv/bin/python -m mypy src/`.
- `.venv/bin/python -m ruff check src/`.
- `env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth` (layer enforcement).
- `scripts/enforce_freeze_guards.py` (or whatever the project's freeze-guard check is) on the new carrier module.

**End-state line count**: `_compose_loop` becomes ≈150–200 lines (the outer driver + per-iteration accumulator updates). `_compose_loop_carriers.py` ≈100 lines. Five new methods ≈ 1,400 lines distributed (matching what they replace, minus duplicate comments). Net: comparable LoC, vastly better structure.

---

## 5. Hidden-coupling callouts

### 5.1 Audit/telemetry ordering — `recorder.record` is a write target of every phase

`recorder` (BufferingRecorder, line 1976) accumulates `ComposerToolInvocation` and `ComposerLLMCall` records across the entire compose call. Every phase reads from and writes to it; it cannot be confined to one helper. The discipline the writer must preserve:

- **Per-tool sub-dispatch** (inside P3): the ordering `recorder.record(finish_*)` → `anti_anchor.record_*(...)` → `llm_messages.append({...})` → `_append_tool_outcome(...)` is repeated across all 8 sub-arms (JSON-decode, non-dict args, canonicalization, cache hit, required-paths, proposal intercept, advisor branches, session-aware, preview-preflight, ToolArgumentError, general success). The writer must diff each arm individually pre-/post-extraction.
- **Convergence raises that consume recorder.invocations** at lines 2064, 2811, 2829 (advisor), 3519, 3531 — these all read the recorder buffer *as-of the raise point*. If the writer accidentally re-orders any `recorder.record(...)` relative to its `raise`, the convergence envelope drops records.
- **`tool_invocations=() if persisted_tool_call_turn else recorder.invocations`** at lines 3524 and 3536 — this is the deduplication discipline that prevents records appearing both in `persist_compose_turn_async` AND in the convergence error's `partial_state`. The writer must keep `persisted_tool_call_turn` flowing as part of `_PersistOutcome` into P5's classify, NOT recompute it.

### 5.2 `state` mutation inside dispatch

Lines 2912 (session-aware) and 3258 (success path) rebind `state`. The closure at lines 3051–3074 (`_do_dispatch`) **captures `state` by default-argument value** specifically to avoid the late-bind bug (the comment at lines 3041–3050 is load-bearing). The writer must NOT change this to a closed-over name; the carrier captures the *post-dispatch* state, but inside the for-loop each iteration sees the latest `state` value via the explicit default-arg pattern.

### 5.3 `llm_messages` is mutated by every phase

| Phase | Lines | Mutation |
|---|---|---|
| P2 (repair branches) | 2086, 2103, 2117(via helper), 2143(via helper) | `.append({"role": "user", ...})` for repair prompts |
| P3 (per-tool) | 2195 (assistant), 2303, 2354, 2402, 2458, 2505, 2617, 2668, 2704, 2774(via advisor helper), 2832, 2875, 2914 (advisor success), 2949(via session-aware helper), 3140, 3301 | `.append({"role": "assistant" or "tool", ...})` |
| P5 (anti-anchor) | 3449 | `.append({"role": "user", ...})` for anchor-break hint |

`llm_messages` cannot be returned in a carrier (it's mutated by every phase including pre-LLM-call repair). Pass it by reference through every helper. **This is intentional** — the carriers do not deep-copy it. The writer should not "try to be clean" by snapshotting it.

### 5.4 Exception paths that bypass `_PersistOutcome`

Multiple raises inside P3 bypass P4 entirely:

- Line 2530: `ComposerServiceError` (too many proposals).
- Line 2811: `ComposerConvergenceError` (advisor timeout exceeds compose deadline).
- Line 2829: convergence on advisor timeout-no-budget.
- Lines 2978–2984: `ComposerRuntimePreflightError.capture` (preview preflight).
- Line 3190: re-raise of `(AssertionError, MemoryError, RecursionError, SystemError)`.
- Line 3192: re-raise of `AuditIntegrityError`.

After extraction, the helper `_dispatch_tool_batch` propagates these exceptions naturally (the `try`/`except` boundaries stay inside the helper or its callees). The driver's only response is to let them through — do NOT wrap the helper call in an additional `try`/`except`.

### 5.5 Test-hook attributes `self._phase3_last_*`

Five test-hook attributes are written inside the current loop body:

| Line | Attribute | Phase that writes it |
|---|---|---|
| 2230 | `self._phase3_last_expected_current_state_id` | P3 (start of each turn) |
| 3314 | `self._phase3_last_tool_outcomes` | P3 (end of dispatch loop) |
| 3380 | `self._phase3_last_redacted_assistant_tool_calls` | P4 (post-redaction) |
| 3381 | `self._phase3_last_redacted_tool_rows` | P4 |
| 3403 | `self._phase3_last_audit_outcome` | P4 (post-persist) |

These reads must stay in the same code path (whatever helper now owns the line). The writer must `grep -rn "_phase3_last_" tests/` and verify each test reads the attribute at the correct compose-loop point (the attributes are *latest-write-wins*; a test that reads after a no-tools return will still see the last tool-call turn's values, which is the current semantics).

### 5.6 BufferingRecorder vs Landscape are different things

The brief uses "flush_audit" as the fourth phase name. This conflates two distinct invariants:

- **BufferingRecorder** (`recorder` in this file): in-memory, accumulates `ComposerToolInvocation` and `ComposerLLMCall` records across the entire compose call. Read by every convergence/crash envelope. Owned by the driver; written from every phase.
- **Landscape** (via `persist_compose_turn_async`): DB-backed audit trail. Written exactly once per tool-call turn (line 3385), under the per-call `_ToolOutcome` discipline.

The `_PersistOutcome` carrier is the boundary for the *Landscape* write, not for the recorder. The recorder is not "flushed" anywhere — it accumulates for the lifetime of the compose call and is read into convergence-error envelopes. **Do not name P4 "flush_audit"** in the final code — the noun is misleading. The name in §2.4 (`_PersistOutcome`) is deliberate.

### 5.7 The `partial_state` discipline on `ComposerPluginCrashError`

The comment block at lines 3196–3247 documents that `plugin_crash = ComposerPluginCrashError.capture(...)` captures `state` (rebound across successful prior tool calls) so the route layer can persist accumulated mutations into composition_states before returning HTTP 500. After extraction, the helper that builds `plugin_crash` (P3, line 3240) reads the *current* `state` — which is correct because state has already been rebound by every prior successful tool call within the same dispatch batch. The writer must NOT move the plugin_crash capture earlier (before the state-rebinding) or later (after `_append_tool_outcome` — it stays bracketed where it is).

---

## 6. Out-of-scope flags

Items discovered during anatomy that the writer should NOT fix in this branch, but should not silently absorb either.

### 6.1 `self._phase3_last_*` test-hook proliferation

Five `self._phase3_last_*` attributes (§5.5) are mutable state on the service instance written from inside the compose loop solely for test observation. This is the pattern this refactor is meant to replace, not extend. After the carriers exist, the carriers *are* the structural assertion surface; the `_phase3_last_*` shims should be deprecated.

**Recommendation**: file a follow-up filigree issue (P3, label `tech-debt`) titled "Replace `_phase3_last_*` test-hook attributes with carrier-based assertion API"; dependency-link as a *successor* of the `_compose_loop` decomposition issue. **Do not delete them in this branch** — that's its own behaviour-preservation risk and the writer's scope is large enough already.

### 6.2 The `from elspeth.web.sessions._persist_payload import _ToolOutcome` inside the function body

Line 2224 is a function-local import of `_ToolOutcome`. The comment cluster around lines 3318–3321 has similar lazy imports for redaction. These look like CLAUDE.md "Shifting the Burden" candidates (lazy import to dodge a layer-import rule). After extraction, the carrier module is the natural place to centralise these imports — moving them top-level there is part of Step 1.

**Recommendation**: in-scope. The carrier module's top-level imports replace these lazy ones. If a layer-import lint fires, that's a *separate* signal — surface to operator, do not paper over with allowlist entries.

### 6.3 The pre-loop `await self._maybe_upsert_skill_markdown_history()` (line 1970)

This is a side-effecting DB write that happens before the carriers exist. It is correctly outside the loop and should stay there. Mentioned only so the writer doesn't try to push it inside P1 "to be tidy" — it must remain a once-per-`_compose_loop`-entry call.

### 6.4 The `from pydantic import ValidationError as PydanticValidationError` shadow at line 2535 vs 3318

Two distinct function-local re-imports of the same name inside the same method. After extraction, both become top-of-module imports in the appropriate helper file. In-scope cleanup.

### 6.5 "Resume paths use the loop" — brief framing is stale

`grep -rn "_compose_loop\|compose_loop" src/elspeth/web/` finds zero `resume`-named entrypoint into `_compose_loop`. The `compose_loop` strings in `sessions/` are `writer_principal="compose_loop"` constants on audit rows — naming, not invocation. The brief's "resume paths use the loop" is most likely a stale carryover from an earlier ELSPETH design; the current code's "resume" semantics are entirely route-layer (replaying chat history) and never re-enter `_compose_loop` mid-iteration. **No resume-related handling is needed in this refactor**; surface to operator if the brief author wants to clarify, but do not invent a resume path the code doesn't have.

---

## Confidence Assessment

- **Loop anatomy (§1)**: High confidence. Every line citation has been verified by direct read or grep.
- **Carrier design (§2)**: Moderate-high. `_DispatchOutcome`'s field set is reverse-engineered from the per-iteration scratch inventory (§1.3 rows A–N); any field I omitted would surface as a missing-argument error during Step 5 extraction. The `freeze_fields` calls on `decoded_args_by_call_id` and `tool_outcomes` are confirmed against `contracts/freeze.py` semantics; `_ToolOutcome`'s own freeze guarantee is verified.
- **Test inventory (§3)**: Moderate. Sampled by file-name pattern + per-file `def test_` list; did not read every test body. Gaps in §3.3 are conservative — writer should `grep` to confirm before assuming a gap is real.
- **Move sequence (§4)**: Moderate. Step 5's diff size (≈1,100 lines) is genuinely large; the sub-split suggestion is necessary if any test breaks mid-extraction. Steps 0/1/2/3/4/6/7 are independently small and safe.
- **Hidden coupling (§5)**: High confidence for the coupling list itself; moderate confidence that I caught everything. The audit/telemetry ordering risk is real and the writer should be told to diff per-arm.

## Risk Assessment

- **Highest risk**: Step 5 (P3 dispatch extraction). 1,100-line diff, eight distinct dispatch sub-arms, audit-record ordering invariants per arm. **Mitigation**: writer splits into Step 5a (pre-dispatch validation arms) and Step 5b (per-tool-arms + success path), runs safety net between each.
- **Second risk**: behaviour preservation of the B-4D-3 last-chance branch. **Mitigation**: Step 0 adds characterization tests if grep shows they're absent. **The writer must not skip Step 0.**
- **Third risk**: the integration latency test (`test_compose_loop_latency_sanity.py::test_per_turn_p95_under_250ms_with_8_tool_calls`). Five extra function call boundaries could plausibly bump p95. **Mitigation**: run this test at end of Step 5 AND Step 6.
- **Layer-import risk**: the carrier module is L3 and imports from L0 (`contracts/freeze.py`), L1 (`core/composer/...`), and L3 (`web/sessions/_persist_payload`, `web/composer/errors`, `web/composer/failed_turn`). All downward. No new layer violations expected, but Step 8's `trust_tier.tier_model` check is the gate.
- **Behaviour-change risk in carrier construction**: `freeze_fields(self, "decoded_args_by_call_id", ...)` deep-freezes the args dicts at carrier construction. If any downstream code (e.g. P4 redaction) mutated those dicts in place under the current implementation, freezing would break it. **The writer must confirm** that `redact_tool_call_arguments` (called in P4's loop at line 3339) treats its input dict as read-only. (Spot-check of `redaction.py` is part of Step 4's prep.)

## Information Gaps

- **Not opened**: `redaction.py::redact_tool_call_arguments` — to confirm in-place mutation absence (see Risk above).
- **Not opened**: `_dispatch_session_aware_tool` body — its `Session-AwareDispatchOutcome.is_discovery / .result` shape is taken on faith from the read context.
- **Not opened**: `BufferingRecorder` source — confirmed its name and the `record/invocations/llm_calls` surface from the call sites, but did not read its class definition.
- **Not investigated**: whether `_compose_loop_carriers.py` collides with any existing file in `tests/` or imports cycles into `_persist_payload.py`. The plan assumes a clean import graph; writer to confirm.
- **Not run**: any test. The plan is structural only.

## Caveats

- The writer must read each test in §3.1 before moving the code that test pins. The one-line summaries above are taxonomic, not specifications.
- The "Mode" declaration above is "Refactor" — public API (`compose()`) unchanged. If during Step 5 the writer finds that an exception class needs to change shape, or that a callsite outside `service.py` reads `self._phase3_last_*` in a way the carriers can't satisfy, the work re-classifies as "Rearchitect" mid-stream — surface to operator before proceeding.
- The freeze-guards CI check (`scripts/cicd/enforce_freeze_guards.py` per CLAUDE.md) must be satisfied for `_DispatchOutcome`. If the project's freeze-guard allowlist mechanism complains about the new dataclass, that is a *signal*, not paperwork — verify the `__post_init__` actually freezes everything, then add the allowlist entry only if the structure is provably correct.
- The brief mentioned "resume paths use the loop". §6.5 contradicts that framing. **Surface this to the operator if the writer is asked to add resume-related coverage** — it would otherwise consume effort against a non-existent code path.
