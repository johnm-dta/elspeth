# Composer Progress Persistence — Phase 3: Compose-Loop Persistence + Tool-Call Cap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the compose loop in `src/elspeth/web/composer/service.py` through Phase 1's `persist_compose_turn` and Phase 2's redaction primitives. Enforce the per-turn tool-call cap. Surface `failed_turn` and `tool_responses_persisted` on error responses. Expose `include_tool_rows=true` on the messages endpoint with audit-grade access logging.

**Architecture:** This is the integration phase. No new schema, no new primitives — just the loop, route handlers, and one new reason code on `ComposerProgressEvent`. The audit-grade transcript view (§6.3) and access-log writes also land here.

**Tech Stack:** Python 3.13, FastAPI / route handlers, structlog, asyncio, pytest with `ChaosLLM` fixture, Hypothesis for property tests, testcontainers for CL-PP-11 regression.

**Spec sections:** §2 (status table), §5.2 (loop shape), §5.5 (failure-mode interaction table), §6.1–§6.3 (route response shape and audit-grade auth), §8.2 (CL-PP-1..13), §8.3 (property test), §11 Phase 3 scope.

---

## File Structure

### Files to modify

- `src/elspeth/web/composer/service.py` — primary surgery. Replace the `_compose_loop` body with the rev-4 shape (§5.2.1). Inject `SessionsService` and `_max_tool_calls_per_turn` via constructor.
- `src/elspeth/web/composer/protocol.py` — add `tool_call_cap_exceeded` reason code; update `ToolArgumentError` docstring at line 299-303 to enumerate all three `except` branches around `execute_tool()` per spec §2.
- `src/elspeth/web/composer/progress.py` — add the new reason code constant.
- `src/elspeth/web/sessions/routes.py` — `_handle_*` route helpers gain the `failed_turn` field; `GET /api/sessions/{sid}/messages` gains the `include_tool_rows` query parameter and audit-grade access-log emission.
- `src/elspeth/web/dependencies.py` (or equivalent) — wire `SessionsService` into the composer service constructor.

### Files to create

- `src/elspeth/web/sessions/audit_access_log.py` — small helper for writing `audit_access_log` rows.
- `tests/integration/pipeline/test_composer_llm_eval_characterization.py` — extend with CL-PP-1..13 (some scenarios already exist; the new ones are CL-PP-9, 10a, 10b, 11, 12, 13).
- `tests/property/web/composer/test_compose_loop_invariants.py` — Hypothesis property test.
- `tests/property/web/composer/strategies.py` — strategy module per spec §8.3.1.
- `tests/integration/web/test_audit_grade_view.py` — exercises `include_tool_rows=true` and asserts the access-log row is written.
- `tests/unit/web/composer/test_tool_call_cap.py` — CL-PP-12.
- `tests/unit/web/composer/test_compose_loop_persistence.py` — happy path and failure paths.

### Files NOT touched in Phase 3

- `src/elspeth/web/sessions/models.py` — schema is finalised in Phase 1.
- `src/elspeth/web/composer/redaction.py` — primitives are finalised in Phase 2.
- Anything under `src/elspeth/web/frontend/` — Phase 4.

---

## Task 1: Add `tool_call_cap_exceeded` reason code

**Files:**
- Modify: `src/elspeth/web/composer/progress.py`
- Test: `tests/unit/web/composer/test_progress_reasons.py` (create or extend)

- [ ] **Step 1: Write the failing test**

```python
"""Tests for the rev-4 tool_call_cap_exceeded reason code (spec §1.4 NFR)."""
from elspeth.web.composer.progress import REASON_CODES, ComposerProgressEvent


def test_tool_call_cap_exceeded_reason_code_exists():
    assert "tool_call_cap_exceeded" in REASON_CODES


def test_progress_event_accepts_new_reason():
    evt = ComposerProgressEvent(
        run_id="r1",
        reason="tool_call_cap_exceeded",
        observed=17,
        cap=16,
    )
    assert evt.reason == "tool_call_cap_exceeded"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_progress_reasons.py -v
```
Expected: FAIL — reason code does not exist.

- [ ] **Step 3: Add the reason code**

In `src/elspeth/web/composer/progress.py` (existing reason codes are at ~lines 63-81 per the spec), add:

```python
REASON_CODES = (
    # ... existing entries from commit 4fce0cae:
    "convergence_wall_clock_timeout",
    "convergence_discovery_budget",
    "convergence_composition_budget",
    "client_cancelled",
    "runtime_preflight_failed",
    # New in rev 4 (spec §1.4 NFR / RSK-13):
    "tool_call_cap_exceeded",
)
```

If `ComposerProgressEvent` validates the reason field (e.g., a `Literal[...]` type), update the literal to include the new value.

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_progress_reasons.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/progress.py tests/unit/web/composer/test_progress_reasons.py
git commit -m "feat(composer): add tool_call_cap_exceeded reason code (composer-progress-persistence phase 3)"
```

---

## Task 2: Update `ToolArgumentError` docstring per spec §2

**Files:**
- Modify: `src/elspeth/web/composer/protocol.py` (around line 291–303)

- [ ] **Step 1: Locate the docstring**

```bash
grep -n "ToolArgumentError\|This is the ONLY exception" src/elspeth/web/composer/protocol.py
```
Expected: line 291 (class) and ~line 299 (docstring).

- [ ] **Step 2: Replace the misleading "ONLY exception" wording**

The rev-3 docstring claimed `ToolArgumentError` is the only exception class caught around `execute_tool()`. Reality is three branches. Update to:

```python
class ToolArgumentError(Exception):
    """Raised by a tool when the LLM-provided arguments are invalid.

    The compose loop catches this exception class around execute_tool()
    and CONTINUES to the next tool call, returning the error as the tool
    response (Tier-3 boundary signal).

    Other exception classes the compose loop catches around execute_tool():
      - (AssertionError, MemoryError, RecursionError, SystemError) at
        service.py:907 — interpreter-level invariant violations.
        Re-raised, NOT continued.
      - Exception at service.py:942 — plugin-bug surface. Wrapped in
        ComposerPluginCrashError.capture(...) from tool_exc, then raised.
        The route layer dispatches via _handle_plugin_crash.

    ToolArgumentError is the only exception class that LEAVES the loop
    iteration alive; the others either re-raise outright or unwind via
    ComposerPluginCrashError.
    """
```

- [ ] **Step 3: No new test required for this docstring change**

The existing service-layer behaviour is unchanged; this is documentation accuracy. The behavioural assertion (that `execute_tool()` exception handling is correctly preserved) is exercised by Tasks 6 onward.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/protocol.py
git commit -m "docs(composer): correct ToolArgumentError docstring to enumerate all three except branches (composer-progress-persistence phase 3)"
```

---

## Task 3: Inject `SessionsService` into the composer service constructor

**Files:**
- Modify: `src/elspeth/web/composer/service.py` — constructor signature
- Modify: `src/elspeth/web/dependencies.py` (or wherever the composer service is constructed at app startup) — pass `SessionsService` and `max_tool_calls_per_turn`
- Test: `tests/unit/web/composer/test_composer_holds_sessions_service.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""Spec §5.1: composer service must hold a SessionsService handle."""
from __future__ import annotations

from unittest.mock import MagicMock


def test_composer_service_accepts_sessions_service():
    from elspeth.web.composer.service import ComposerService
    sessions = MagicMock()  # Phase-3 unit test only; full integration tests use real service
    composer = ComposerService(sessions_service=sessions, max_tool_calls_per_turn=16)
    assert composer.sessions_service is sessions
    assert composer._max_tool_calls_per_turn == 16


def test_composer_service_default_cap_matches_nfr():
    from elspeth.web.composer.service import ComposerService
    sessions = MagicMock()
    composer = ComposerService(sessions_service=sessions)
    assert composer._max_tool_calls_per_turn == 16  # spec §1.4 NFR default
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_composer_holds_sessions_service.py -v
```
Expected: FAIL.

- [ ] **Step 3: Update the constructor**

In `src/elspeth/web/composer/service.py`:

```python
class ComposerService:
    def __init__(
        self,
        # ... existing dependencies ...
        sessions_service: SessionsService,
        max_tool_calls_per_turn: int = 16,
    ) -> None:
        # ... existing init body ...
        self.sessions_service = sessions_service
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
```

In `src/elspeth/web/dependencies.py` (or wherever `ComposerService` is constructed), wire it:

```python
composer_service = ComposerService(
    # ... existing args ...
    sessions_service=sessions_service,  # already exists in DI graph
    max_tool_calls_per_turn=settings.composer.max_tool_calls_per_turn or 16,
)
```

If the settings layer has no `composer.max_tool_calls_per_turn` field, add it. Default 16.

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_composer_holds_sessions_service.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py src/elspeth/web/dependencies.py tests/unit/web/composer/test_composer_holds_sessions_service.py
git commit -m "feat(composer): inject SessionsService and max_tool_calls_per_turn (composer-progress-persistence phase 3)"
```

---

## Task 4: Tool-call cap enforcement (CL-PP-12)

**Files:**
- Modify: `src/elspeth/web/composer/service.py` — at the top of `_compose_loop`'s per-turn body
- Test: `tests/unit/web/composer/test_tool_call_cap.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""CL-PP-12 (spec §8.2): tool-call cap exceeded fails fast before any
tool execution; counter increments; reason code propagates."""
from __future__ import annotations

import pytest

from elspeth.web.composer.protocol import ComposerConvergenceError


@pytest.mark.asyncio
async def test_cap_exceeded_raises_before_tool_execution(composer_service_with_chaos):
    """ChaosLLM emits an assistant turn with 17 tool calls; cap is 16.
    Loop must raise BEFORE any execute_tool() runs."""
    composer, chaos = composer_service_with_chaos
    composer._max_tool_calls_per_turn = 16

    chaos.queue_assistant_turn(
        content="",
        tool_calls=[
            {"id": f"tc_{i}", "function": {"name": "fake_tool", "arguments": "{}"}}
            for i in range(17)
        ],
    )

    with pytest.raises(ComposerConvergenceError) as exc_info:
        await composer.run("session_x", "user prompt")
    assert exc_info.value.reason == "tool_call_cap_exceeded"
    # Counter incremented:
    assert composer.sessions_service._telemetry.tool_call_cap_exceeded_total.observed_value >= 1
    # No tool executed:
    assert chaos.execute_tool_calls == 0
```

(`composer_service_with_chaos` is a fixture that wires a real composer service with the project's `ChaosLLM` LLM mock and a `ChaosTool` registry. Define it in `tests/conftest.py` or a phase-local conftest if not already present. The fixture uses an in-memory SQLite engine for `SessionsService`.)

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_tool_call_cap.py -v
```
Expected: FAIL — cap enforcement not yet in the loop.

- [ ] **Step 3: Add cap enforcement at the top of each turn**

In `src/elspeth/web/composer/service.py`'s `_compose_loop`, immediately after obtaining `assistant_message` from the LLM:

```python
# Spec §1.4 NFR / RSK-13: defend against LLM amplification.
if len(assistant_message.tool_calls or ()) > self._max_tool_calls_per_turn:
    self.sessions_service._telemetry.tool_call_cap_exceeded_total.add(1)
    raise ComposerConvergenceError.capture(
        state,
        reason="tool_call_cap_exceeded",
        observed=len(assistant_message.tool_calls or ()),
        cap=self._max_tool_calls_per_turn,
    )
```

(If `ComposerConvergenceError.capture()` does not accept arbitrary kwargs today, extend its signature in `protocol.py` to accept and store them on the exception instance.)

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_tool_call_cap.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/service.py src/elspeth/web/composer/protocol.py tests/unit/web/composer/test_tool_call_cap.py
git commit -m "feat(composer): per-turn tool-call cap with reason code (composer-progress-persistence phase 3)"
```

---

## Task 5: Tool-execution accumulation loop (Step 1 of §5.2.1)

**Files:**
- Modify: `src/elspeth/web/composer/service.py`
- Test: `tests/unit/web/composer/test_compose_loop_persistence.py` (create)

> **Phase 1 cross-phase contract reminder — `raw_content` plumbing (B2 / B2-followup-1).**
>
> Phase 1's `SessionServiceImpl.persist_compose_turn` accepts an optional
> `raw_content: str | None = None` kwarg (added per B2 from the Phase 1
> plan-review synthesis). The default is `None` so Phase 1's tests pass
> in isolation, but **the compose loop is the canonical caller and MUST
> supply `raw_content` at every call site** — failing to do so silently
> defeats the redaction-attribution invariant: the audit-attribution
> column would receive `None` for every compose-turn row and the
> pre-redaction LLM output would be unrecoverable from the audit trail.
>
> Spec §5.2.1's call-site example (lines ~895–904) **omits**
> `raw_content` — that example was authored before B2 was identified
> and is now stale. Phase 1 filed a Filigree spec-drift observation
> (see Phase 1 plan Task 19 Step 5/6); the spec amendment is doc-only
> catch-up. **Do not** "replicate spec §5.2.1 verbatim" without adding
> `raw_content` to the call.
>
> Source-of-truth for the value: routes 1754 and 2157 currently pass
> `raw_content=result.raw_assistant_content` to `add_message` for
> assistant rows. Inside `_compose_loop`, the equivalent expression
> per turn is `assistant_message.content or ""` (the verbatim LLM
> output for that turn — no per-turn preflight redaction is applied
> to the natural-language assistant content at the rev-4 design layer,
> so raw and visible content coincide; we still pass the value
> explicitly so a future redaction step is structurally captured by
> the audit column rather than silently coinciding with `content`).
> Step 4 below makes this plumbing explicit; Step 1's happy-path test
> asserts the column lands non-`NULL`.

- [ ] **Step 1: Write the failing test**

```python
"""Compose-loop persistence: happy path with three tool calls.
Spec §5.2.1 / CL-PP-1 / CL-PP-2 + B2-followup-1 raw_content
plumbing assertion."""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_three_tool_calls_persist_one_assistant_three_tool_rows(
    composer_service_with_chaos,
):
    composer, chaos = composer_service_with_chaos
    chaos.queue_assistant_turn(
        content="first turn assistant text",
        tool_calls=[
            {"id": f"tc_{i}", "function": {"name": "fake_tool", "arguments": "{}"}}
            for i in range(3)
        ],
    )
    for i in range(3):
        chaos.queue_tool_response(f"tc_{i}", {"ok": True, "i": i})
    chaos.queue_assistant_turn(content="done", tool_calls=[])  # terminate

    await composer.run("session_y", "user prompt")

    with composer.sessions_service._engine.begin() as conn:
        rows = conn.execute(__import__("sqlalchemy").text(
            "SELECT role, sequence_no, tool_call_id, content, raw_content "
            "FROM chat_messages WHERE session_id='session_y' AND role IN ('assistant','tool') "
            "ORDER BY sequence_no"
        )).fetchall()
    # 1 user (already in fixture) + 2 assistant + 3 tool rows visible to this query
    # (fixture created the user message at sequence_no=1).
    assistant_rows = [r for r in rows if r.role == "assistant"]
    tool_rows = [r for r in rows if r.role == "tool"]
    assert len(assistant_rows) == 2  # one with 3 tool calls, one terminating
    assert len(tool_rows) == 3
    assert {r.tool_call_id for r in tool_rows} == {"tc_0", "tc_1", "tc_2"}

    # B2-followup-1: every compose-turn assistant row carries raw_content
    # (the pre-redaction LLM output for audit attribution). Phase 1's
    # persist_compose_turn defaults raw_content to None; the compose loop
    # MUST plumb assistant_message.content explicitly so this column is
    # never silently NULL in the audit trail.
    assert assistant_rows[0].raw_content == "first turn assistant text"
    assert assistant_rows[1].raw_content == "done"
    # Tool rows carry raw_content=None — redaction-attribution applies
    # only to LLM-authored content (Phase 1 plan Task 11 docstring).
    assert all(r.raw_content is None for r in tool_rows)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v
```
Expected: FAIL — loop does not yet persist anything per turn.

- [ ] **Step 3: Implement the Step-1 accumulation in `_compose_loop`**

In `src/elspeth/web/composer/service.py`, replace the existing tool-execution block with the rev-4 shape from spec §5.2.1. The full body is given there; replicate it verbatim into the loop, replacing the in-memory append-to-`llm_messages` pattern with the `_ToolOutcome` accumulator + `persist_compose_turn` dispatch.

Specifically the per-turn body becomes:

```python
# (cap check from Task 4 already runs here)

# Step 1 — execute every tool call in async land, accumulating outcomes.
tool_outcomes: list[_ToolOutcome] = []
plugin_crash: BaseException | None = None
initial_version = state.version

for tool_call in assistant_message.tool_calls or ():
    pre_version = state.version
    try:
        response = await execute_tool(tool_call, state)
        tool_outcomes.append(_ToolOutcome(
            call=tool_call, response=response,
            error_class=None, error_message=None,
            pre_version=pre_version, post_version=state.version,
        ))
    except ToolArgumentError as exc:
        tool_outcomes.append(_ToolOutcome(
            call=tool_call, response=None,
            error_class="ToolArgumentError", error_message=str(exc),
            pre_version=pre_version, post_version=state.version,
        ))
        # service.py:867 — continue.
    except (AssertionError, MemoryError, RecursionError, SystemError):
        # service.py:907 — re-raise.
        raise
    except Exception as tool_exc:
        # service.py:942-980 — capture and break.
        tool_outcomes.append(_ToolOutcome(
            call=tool_call, response=None,
            error_class=type(tool_exc).__name__,
            error_message=str(tool_exc),
            pre_version=pre_version, post_version=state.version,
        ))
        plugin_crash = ComposerPluginCrashError.capture(
            tool_exc, state=state, initial_version=initial_version,
        )
        break

# (Step 2: redact + dispatch; see Task 6.)
# (Step 3: handle audit outcome and re-raise plugin_crash; see Task 6.)
```

(Add imports for `_ToolOutcome` from `elspeth.web.sessions._persist_payload`.)

- [ ] **Step 4: Add Steps 2 and 3 from spec §5.2.1 — with the B2-followup-1 `raw_content` amendment**

Continuing in the same loop body, add Step 2 (redact + dispatch) and Step 3 (handle outcome). Full source in spec §5.2.1; replicate **with one mandatory amendment**: the `persist_compose_turn(...)` call site at spec lines ~895–904 omits `raw_content`. That omission is stale spec text (§5.2.1 was authored before Phase 1's B2 fix added `raw_content: str | None = None` to the primitive's signature). **You MUST add `raw_content=assistant_message.content or ""` to the kwargs.** See the Phase 1 cross-phase contract reminder block at the top of this Task for the rationale; not plumbing this kwarg silently sets `chat_messages.raw_content = NULL` for every compose-turn assistant row and breaks the redaction-attribution invariant.

Highlights:

- `redact_tool_call(tc, lookup_tool_class(tc.function.name))` for each assistant tool call.
- Build `_RedactedToolRow` for each tool outcome using `apply_response_redaction` for legacy-policy tools and `apply_sensitive_redaction` for type-driven tools (or, simpler: a single helper `redact_tool_response(outcome, tool_class)` that dispatches internally — implementer's choice; the spec is agnostic).
- The `persist_compose_turn` call MUST take this shape (deviating from spec §5.2.1 lines ~895–904 by adding the `raw_content` kwarg per B2-followup-1):

  ```python
  audit_outcome = await self._run_sync(
      self.sessions_service.persist_compose_turn,
      session_id=session_id,
      assistant_content=assistant_message.content or "",
      raw_content=assistant_message.content or "",  # B2-followup-1: pre-redaction
                                                    # LLM output for audit
                                                    # attribution. Phase 1's
                                                    # persist_compose_turn defaults
                                                    # this to None; the compose loop
                                                    # is the canonical caller and
                                                    # MUST supply it. Spec §5.2.1's
                                                    # call-site example omits this
                                                    # kwarg — that example is stale
                                                    # (see Phase 1 plan Task 19's
                                                    # B2 spec-drift observation).
                                                    # See routes.py lines 1754, 2157
                                                    # for the legacy add_message
                                                    # equivalents passing
                                                    # raw_content=result.raw_assistant_content.
      redacted_assistant_tool_calls=redacted_assistant_tool_calls,
      redacted_tool_rows=redacted_tool_rows,
      parent_composition_state_id=current_state_id,
      writer_principal="compose_loop",
      plugin_crash_pending=plugin_crash is not None,
  )
  ```

- Dispatch on `audit_outcome.tier1_violation` → raise; `plugin_crash is not None` → raise.

- [ ] **Step 5: Verify `raw_content` plumbing is wired (B2-followup-1 gate)**

Before running the persistence test, confirm by inspection that the
`persist_compose_turn(...)` call inside `_compose_loop` passes
`raw_content=assistant_message.content or ""` (or an equivalent
expression that resolves to the verbatim LLM-authored assistant text
for the turn). Run:

```bash
grep -n "persist_compose_turn(" src/elspeth/web/composer/service.py
grep -n "raw_content=" src/elspeth/web/composer/service.py
```

Expected: every `persist_compose_turn(` call inside `_compose_loop`
appears in the `raw_content=` grep with a non-`None` source expression.
If any call site omits `raw_content` or passes `None` explicitly, the
audit-attribution invariant is broken — fix before proceeding.

- [ ] **Step 6: Run test to verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v
```
Expected: PASS, including the `raw_content` assertions added per B2-followup-1.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/composer/service.py tests/unit/web/composer/test_compose_loop_persistence.py
git commit -m "feat(composer): rev-4 _compose_loop per-turn persistence dispatch (composer-progress-persistence phase 3)

Plumbs raw_content=assistant_message.content into persist_compose_turn
per B2-followup-1 from the Phase 1 plan-review closure (audit-attribution
invariant — Phase 1's persist_compose_turn defaults raw_content to None,
the compose loop is the canonical caller and MUST supply it). Spec
§5.2.1's call-site example omits raw_content; that omission is stale
spec text superseded by Phase 1's B2 fix."

---

## Task 6: CL-PP-1 — convergence error mid-loop preserves partial state

**Files:**
- Modify: `tests/integration/pipeline/test_composer_llm_eval_characterization.py`
- Modify: `src/elspeth/web/sessions/routes.py` — `_handle_convergence_error` adds `failed_turn` field

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_cl_pp_1_convergence_mid_loop_preserves_breadcrumbs(
    chaos_composer_with_real_db,
):
    """CL-PP-1: budget exhaustion at turn N leaves all completed-turn rows
    in chat_messages; failed_turn surfaces the count discrepancy."""
    composer, chaos, db = chaos_composer_with_real_db
    composer._budget_remaining_turns = 2  # force exhaustion at turn 3

    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_t0_a", "function": {"name": "fake_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_response("tc_t0_a", {"ok": True})
    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_t1_a", "function": {"name": "fake_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_response("tc_t1_a", {"ok": True})
    # Turn 2 exhausts budget mid-execution.

    response = await composer.run_via_route("session_cv", "go")
    assert response.status_code == 422
    body = response.json()
    assert body["reason"] in ("convergence_discovery_budget", "convergence_composition_budget")
    assert "partial_state" in body
    assert "failed_turn" in body
    assert body["failed_turn"]["tool_calls_attempted"] >= 0
    assert body["failed_turn"]["tool_responses_persisted"] >= 0

    # Verify chat_messages reflects what landed.
    with db.begin() as conn:
        count = conn.execute(__import__("sqlalchemy").text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='session_cv' AND role='tool'"
        )).scalar()
    assert count == body["failed_turn"]["tool_responses_persisted"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_llm_eval_characterization.py::test_cl_pp_1_convergence_mid_loop_preserves_breadcrumbs -v
```
Expected: FAIL — `failed_turn` field is not in the response body.

- [ ] **Step 3: Add `failed_turn` to `_handle_convergence_error`**

In `src/elspeth/web/sessions/routes.py`, find the three `_handle_*` helpers and add the `failed_turn` field per spec §6.1:

```python
def _handle_convergence_error(exc, ..., sessions_service, session_id, last_user_message_id):
    response_body = {
        # ... existing fields ...
        "failed_turn": _build_failed_turn(
            sessions_service=sessions_service,
            session_id=session_id,
            last_user_message_id=last_user_message_id,
            assistant_msg_id=getattr(exc, "assistant_msg_id", None),
        ),
    }
    return JSONResponse(status_code=422, content=response_body)


def _build_failed_turn(*, sessions_service, session_id, last_user_message_id, assistant_msg_id):
    """Compute failed_turn fields for the response body. Reads chat_messages
    AFTER the compose loop's persist completed (the shielded write is no
    longer used in rev 4 — single sync block ensures the writes have
    committed before this code runs). Spec §6.1 read-consistency note."""
    if assistant_msg_id is None:
        return None
    with sessions_service._engine.begin() as conn:
        attempted_row = conn.execute(text(
            "SELECT tool_calls FROM chat_messages WHERE id = :aid"
        ), {"aid": assistant_msg_id}).first()
        attempted = len(attempted_row.tool_calls or []) if attempted_row else 0
        persisted = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages "
            "WHERE parent_assistant_id = :aid AND role = 'tool'"
        ), {"aid": assistant_msg_id}).scalar() or 0
    return {
        "assistant_message_id": assistant_msg_id,
        "tool_calls_attempted": attempted,
        "tool_responses_persisted": persisted,
        "transcript_url": (
            f"/api/sessions/{session_id}/messages"
            f"?since={last_user_message_id}&include_tool_rows=true"
        ),
    }
```

Apply the same `failed_turn` block to `_handle_plugin_crash` and `_handle_runtime_preflight_failure`.

- [ ] **Step 4: Run test to verify pass**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_llm_eval_characterization.py::test_cl_pp_1_convergence_mid_loop_preserves_breadcrumbs -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/sessions/routes.py tests/integration/pipeline/test_composer_llm_eval_characterization.py
git commit -m "feat(routes): failed_turn field on _handle_* responses (composer-progress-persistence phase 3)"
```

---

## Task 7: CL-PP-2 — plugin crash mid-loop persists rows + raises

**Files:**
- Modify: `tests/integration/pipeline/test_composer_llm_eval_characterization.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_cl_pp_2_plugin_crash_persists_rows_and_raises(chaos_composer_with_real_db):
    """CL-PP-2: plugin crash on second of three tool calls. Assert
    assistant + 1 normal tool row + 1 error tool row land atomically;
    ComposerPluginCrashError propagates AFTER audit; 500 with partial_state."""
    composer, chaos, db = chaos_composer_with_real_db
    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_a", "function": {"name": "fake_tool", "arguments": "{}"}},
        {"id": "tc_b", "function": {"name": "boom_tool", "arguments": "{}"}},
        {"id": "tc_c", "function": {"name": "fake_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_response("tc_a", {"ok": True})
    chaos.queue_tool_raise("tc_b", RuntimeError("plugin bug"))
    # tc_c is never reached: the loop breaks after tc_b's exception.

    response = await composer.run_via_route("session_pc", "go")
    assert response.status_code == 500
    body = response.json()
    assert "partial_state" in body
    assert body["failed_turn"]["tool_calls_attempted"] == 3
    assert body["failed_turn"]["tool_responses_persisted"] == 2  # tc_a success + tc_b error

    with db.begin() as conn:
        rows = conn.execute(text(
            "SELECT tool_call_id, content FROM chat_messages "
            "WHERE session_id='session_pc' AND role='tool' ORDER BY sequence_no"
        )).fetchall()
    assert [r.tool_call_id for r in rows] == ["tc_a", "tc_b"]
    assert "ok" in rows[0].content
    assert "RuntimeError" in rows[1].content
```

- [ ] **Step 2: Run test**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_llm_eval_characterization.py::test_cl_pp_2_plugin_crash_persists_rows_and_raises -v
```
Expected: PASS (Task 5's loop already implements the break-on-exception + capture behaviour). If not, fix.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/pipeline/test_composer_llm_eval_characterization.py
git commit -m "test(integration): CL-PP-2 plugin crash mid-loop (composer-progress-persistence phase 3)"
```

---

## Task 8: CL-PP-3 through CL-PP-8 (existing scenarios)

The rev-3 spec already lists CL-PP-3 (timeout), CL-PP-4a/b/c (DB write fails on assistant row / tool row variants), CL-PP-5 (summarizer raises), CL-PP-6 (cross-session leakage), CL-PP-7 (mid-loop cancellation race), CL-PP-8 (duplicate tool_call_id). Walk through each:

- [ ] **Step 1: Implement CL-PP-3 (wall-clock timeout in tool execution)**

```python
@pytest.mark.asyncio
async def test_cl_pp_3_wall_clock_timeout_during_tool_execution(chaos_composer_with_real_db):
    composer, chaos, db = chaos_composer_with_real_db
    composer._wall_clock_budget_s = 0.5  # tiny budget

    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_slow", "function": {"name": "slow_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_delay("tc_slow", 5.0)  # exceeds budget

    response = await composer.run_via_route("session_to", "go")
    assert response.status_code == 422
    body = response.json()
    assert body["reason"] == "convergence_wall_clock_timeout"
    assert "partial_state" in body
```

Run, verify pass, commit.

- [ ] **Step 2: Implement CL-PP-4a (DB fails on first INSERT, no plugin crash → Tier-1)**

```python
@pytest.mark.asyncio
async def test_cl_pp_4a_db_fails_no_plugin_crash_tier1_violation(
    chaos_composer_with_real_db, monkeypatch,
):
    composer, chaos, db = chaos_composer_with_real_db

    def boom(self, *args, **kwargs):
        from sqlalchemy.exc import OperationalError
        raise OperationalError("simulated", {}, Exception())

    monkeypatch.setattr(
        composer.sessions_service.__class__, "_insert_chat_message", boom,
    )
    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_a", "function": {"name": "fake_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_response("tc_a", {"ok": True})

    from sqlalchemy.exc import OperationalError
    with pytest.raises(OperationalError):
        await composer.run("session_t1", "go")
    assert composer.sessions_service._telemetry.tool_row_tier1_violation_total.observed_value >= 1
```

- [ ] **Step 3: Implement CL-PP-4c (DB fails during plugin crash unwind)**

```python
@pytest.mark.asyncio
async def test_cl_pp_4c_db_fails_during_plugin_crash_logs_and_raises_plugin_exc(
    chaos_composer_with_real_db, monkeypatch,
):
    composer, chaos, db = chaos_composer_with_real_db
    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_boom", "function": {"name": "boom_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_raise("tc_boom", RuntimeError("plugin bug"))

    def boom(self, *args, **kwargs):
        from sqlalchemy.exc import OperationalError
        raise OperationalError("simulated", {}, Exception())
    monkeypatch.setattr(
        composer.sessions_service.__class__, "_insert_chat_message", boom,
    )

    from elspeth.web.composer.protocol import ComposerPluginCrashError
    with pytest.raises(ComposerPluginCrashError):
        await composer.run("session_t1c", "go")
    counter = composer.sessions_service._telemetry.tool_row_persist_failed_during_unwind_total
    assert counter.observed_value >= 1
```

- [ ] **Step 4: CL-PP-5 (summarizer raises — already covered by Phase 2 unit; reproduce end-to-end)**

```python
@pytest.mark.asyncio
async def test_cl_pp_5_summarizer_raises_falls_back_via_route(chaos_composer_with_real_db):
    """Summarizer that raises produces fallback sentinel in persisted content;
    counter increments; route returns 200 (not an error path)."""
    # Configure the chaos tool to use a summarizer that raises.
    # Detailed setup depends on ChaosTool fixture API — see tests/conftest.py.
    ...
```

- [ ] **Step 5: CL-PP-6, CL-PP-7, CL-PP-8** — implement each per spec §8.2 wording.

- [ ] **Step 6: After each scenario above, commit**

```bash
git add tests/integration/pipeline/test_composer_llm_eval_characterization.py
git commit -m "test(integration): CL-PP-{N} (composer-progress-persistence phase 3)"
```

---

## Task 9: CL-PP-9 mixed-redaction integration test

**Files:**
- Modify: `tests/integration/pipeline/test_composer_llm_eval_characterization.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_cl_pp_9_mixed_redaction_persists_correctly(chaos_composer_with_real_db):
    """Spec §8.2 CL-PP-9: a tool with both Sensitive[T] fields AND
    non-sensitive fields persists redacted values for the former and
    byte-identical values for the latter."""
    composer, chaos, db = chaos_composer_with_real_db
    chaos.queue_assistant_turn(content="", tool_calls=[
        {
            "id": "tc_mix",
            "function": {
                "name": "mixed_tool",
                "arguments": '{"path": "/secret/path", "label": "ok"}',
            },
        },
    ])
    chaos.queue_tool_response("tc_mix", {"ok": True})
    chaos.queue_assistant_turn(content="done", tool_calls=[])

    await composer.run("session_mix", "go")
    with db.begin() as conn:
        row = conn.execute(text(
            "SELECT tool_calls FROM chat_messages "
            "WHERE session_id='session_mix' AND role='assistant' "
            "ORDER BY sequence_no LIMIT 1"
        )).first()
    args = row.tool_calls[0]["function"]["arguments"]
    assert args["path"] != "/secret/path"  # redacted
    assert args["label"] == "ok"           # byte-identical
```

(Requires `mixed_tool` to be defined in the chaos test registry with appropriate `Sensitive[T]` annotation on `path`.)

- [ ] **Step 2: Run, ensure pass, commit**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_llm_eval_characterization.py::test_cl_pp_9_mixed_redaction_persists_correctly -v
git add tests/integration/pipeline/test_composer_llm_eval_characterization.py
git commit -m "test(integration): CL-PP-9 mixed redaction (composer-progress-persistence phase 3)"
```

---

## Task 10: CL-PP-10a/b INSERT-OK-COMMIT-FAIL

**Files:**
- Modify: `tests/integration/pipeline/test_composer_llm_eval_characterization.py`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_cl_pp_10a_commit_fails_no_plugin_crash(chaos_composer_with_real_db, monkeypatch):
    """CL-PP-10a (spec §8.2): INSERT succeeds, COMMIT fails, no plugin crash.
    Result: Tier-1 violation, counter increments, no rows visible after rollback."""
    composer, chaos, db = chaos_composer_with_real_db
    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_a", "function": {"name": "fake_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_response("tc_a", {"ok": True})

    # SQLAlchemy event hook: raise on commit.
    from sqlalchemy import event
    from sqlalchemy.exc import OperationalError

    def fail_commit(conn):
        raise OperationalError("simulated commit failure", {}, Exception())

    event.listen(db, "commit", fail_commit, once=True)

    with pytest.raises(OperationalError):
        await composer.run("session_10a", "go")
    counter = composer.sessions_service._telemetry.tool_row_tier1_violation_total
    assert counter.observed_value >= 1

    # No rows visible — rollback successful.
    with db.begin() as conn:
        rows = conn.execute(text(
            "SELECT COUNT(*) FROM chat_messages WHERE session_id='session_10a'"
        )).scalar()
    assert rows == 0


@pytest.mark.asyncio
async def test_cl_pp_10b_commit_fails_with_plugin_crash(chaos_composer_with_real_db, monkeypatch):
    """CL-PP-10b: INSERT succeeds, COMMIT fails, plugin crash in flight.
    Result: unwind_audit_failed counter increments, log entry, plugin
    exception propagates."""
    composer, chaos, db = chaos_composer_with_real_db
    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_boom", "function": {"name": "boom_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_raise("tc_boom", RuntimeError("plugin bug"))

    from sqlalchemy import event
    from sqlalchemy.exc import OperationalError

    def fail_commit(conn):
        raise OperationalError("simulated", {}, Exception())
    event.listen(db, "commit", fail_commit, once=True)

    from elspeth.web.composer.protocol import ComposerPluginCrashError
    with pytest.raises(ComposerPluginCrashError):
        await composer.run("session_10b", "go")
    counter = composer.sessions_service._telemetry.tool_row_persist_failed_during_unwind_total
    assert counter.observed_value >= 1
```

- [ ] **Step 2: Run, ensure pass, commit**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_llm_eval_characterization.py -v -k cl_pp_10
git add tests/integration/pipeline/test_composer_llm_eval_characterization.py
git commit -m "test(integration): CL-PP-10a/b INSERT-ok-COMMIT-fail (composer-progress-persistence phase 3)"
```

---

## Task 11: CL-PP-13 unknown-response-key fail-closed

**Files:**
- Modify: `tests/integration/pipeline/test_composer_llm_eval_characterization.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_cl_pp_13_unknown_response_key_fail_closed(chaos_composer_with_real_db):
    """CL-PP-13: legacy-escape-valve tool returns a key not in known_response_keys.
    The unknown key's value is replaced with the redacted-unknown-key sentinel
    in the persisted content."""
    composer, chaos, db = chaos_composer_with_real_db
    # A legacy tool with known_response_keys=("ok",) returns {"ok": True, "leaked": "AKIA-X"}
    chaos.queue_assistant_turn(content="", tool_calls=[
        {"id": "tc_legacy", "function": {"name": "legacy_drift_tool", "arguments": "{}"}},
    ])
    chaos.queue_tool_response("tc_legacy", {"ok": True, "leaked": "AKIA-X"})
    chaos.queue_assistant_turn(content="done", tool_calls=[])

    await composer.run("session_drift", "go")
    with db.begin() as conn:
        row = conn.execute(text(
            "SELECT content FROM chat_messages "
            "WHERE session_id='session_drift' AND role='tool'"
        )).first()
    import json
    persisted = json.loads(row.content)
    assert persisted["ok"] is True
    assert persisted["leaked"].startswith("<redacted-unknown-key:")
    assert "AKIA-X" not in row.content
    counter = composer.sessions_service._telemetry.unknown_response_key_total
    assert counter.observed_value >= 1
```

- [ ] **Step 2: Run, ensure pass, commit**

```bash
.venv/bin/python -m pytest tests/integration/pipeline/test_composer_llm_eval_characterization.py::test_cl_pp_13_unknown_response_key_fail_closed -v
git add tests/integration/pipeline/test_composer_llm_eval_characterization.py
git commit -m "test(integration): CL-PP-13 unknown-response-key fail-closed (composer-progress-persistence phase 3)"
```

---

## Task 12: `include_tool_rows` query parameter + audit-grade access logging

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py` — `GET /api/sessions/{sid}/messages`
- Create: `src/elspeth/web/sessions/audit_access_log.py` — small write helper
- Test: `tests/integration/web/test_audit_grade_view.py` (create)

- [ ] **Step 1: Write the failing test**

```python
"""Spec §6.3 audit-grade view + audit_access_log emission."""
from __future__ import annotations

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_include_tool_rows_default_false(chaos_composer_with_real_db, http_client):
    """By default, the messages endpoint hides tool rows (live chat behaviour)."""
    composer, chaos, db = chaos_composer_with_real_db
    # ... seed a session with tool rows ...
    response = await http_client.get(f"/api/sessions/session_x/messages")
    assert response.status_code == 200
    rows = response.json()["messages"]
    assert all(r["role"] != "tool" for r in rows)


@pytest.mark.asyncio
async def test_include_tool_rows_true_emits_access_log(chaos_composer_with_real_db, http_client):
    """include_tool_rows=true returns tool rows AND emits an audit_access_log row."""
    composer, chaos, db = chaos_composer_with_real_db
    # ... seed a session ...
    response = await http_client.get(
        f"/api/sessions/session_x/messages?include_tool_rows=true"
    )
    assert response.status_code == 200
    rows = response.json()["messages"]
    assert any(r["role"] == "tool" for r in rows)

    with db.begin() as conn:
        log_rows = conn.execute(text(
            "SELECT requesting_principal, request_path, query_args, writer_principal "
            "FROM audit_access_log WHERE session_id='session_x'"
        )).fetchall()
    assert len(log_rows) == 1
    assert log_rows[0].writer_principal == "audit_grade_view"
    assert log_rows[0].request_path == "/api/sessions/session_x/messages"
    assert log_rows[0].query_args["include_tool_rows"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/integration/web/test_audit_grade_view.py -v
```
Expected: FAIL.

- [ ] **Step 3: Implement the access-log helper**

Create `src/elspeth/web/sessions/audit_access_log.py`:

```python
"""Audit-grade access-log writer (spec §6.3)."""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import insert
from sqlalchemy.engine import Connection

from . import models


def write_access_log(
    conn: Connection,
    *,
    session_id: str,
    requesting_principal: str,
    request_path: str,
    query_args: Mapping[str, Any],
    ip_address: str | None,
) -> str:
    """Write one row to audit_access_log. Caller supplies the connection
    (typically the same connection used to read the audit-grade rows in
    the same transaction)."""
    log_id = str(uuid.uuid4())
    conn.execute(insert(models.audit_access_log_table).values(
        id=log_id,
        timestamp=datetime.now(timezone.utc),
        session_id=session_id,
        requesting_principal=requesting_principal,
        request_path=request_path,
        query_args=dict(query_args),
        ip_address=ip_address,
        writer_principal="audit_grade_view",
    ))
    return log_id
```

- [ ] **Step 4: Wire the route handler**

In `src/elspeth/web/sessions/routes.py`, find the `GET /api/sessions/{sid}/messages` handler and update:

```python
@router.get("/api/sessions/{session_id}/messages")
async def list_messages(
    session_id: str,
    since: str | None = None,
    include_tool_rows: bool = False,
    request: Request,
    user: User = Depends(current_user),
    sessions_service: SessionsService = Depends(get_sessions_service),
):
    # 1. Existing session-ownership check.
    _ensure_session_owned_by(user, session_id, sessions_service)

    # 2. If audit-grade view requested, emit access-log row in the same
    #    transaction as the read.
    with sessions_service._engine.begin() as conn:
        if include_tool_rows:
            from elspeth.web.sessions.audit_access_log import write_access_log
            write_access_log(
                conn,
                session_id=session_id,
                requesting_principal=user.principal,
                request_path=str(request.url.path),
                query_args={"include_tool_rows": True, "since": since},
                ip_address=request.client.host if request.client else None,
            )
            sessions_service._telemetry.audit_grade_view_total.add(1)

        # 3. Query messages (existing logic, plus include_tool_rows filter).
        query = (
            "SELECT id, role, content, tool_calls, tool_call_id, "
            "parent_assistant_id, sequence_no, created_at "
            "FROM chat_messages WHERE session_id = :sid"
        )
        if not include_tool_rows:
            query += " AND role IN ('user', 'assistant', 'system')"
        if since is not None:
            query += " AND id > :since"  # adjust to project's existing cursor convention
        query += " ORDER BY sequence_no ASC"

        rows = conn.execute(text(query), {"sid": session_id, "since": since}).fetchall()

    return {"messages": [_serialise_row(r) for r in rows]}
```

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/test_audit_grade_view.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/sessions/audit_access_log.py src/elspeth/web/sessions/routes.py tests/integration/web/test_audit_grade_view.py
git commit -m "feat(routes): include_tool_rows query parameter with audit-grade access logging (composer-progress-persistence phase 3)"
```

---

## Task 13: Hypothesis property test for INV-AUDIT-AHEAD

**Files:**
- Create: `tests/property/web/composer/strategies.py`
- Create: `tests/property/web/composer/test_compose_loop_invariants.py`

This task implements the strategies and post-conditions in spec §8.3.

- [ ] **Step 1: Strategy module**

Create `tests/property/web/composer/strategies.py`. Implement each strategy enumerated in spec §8.3.1:

- `st_tool_call`: well-formed `ToolCall` instances; 70% well-formed, 30% sensitive-marked.
- `st_argument_dict`: dicts with string/int/bool/nested-dict to depth 3.
- `st_redaction_policy`: tuples of policy fields.
- `st_failure_injection_point`: enum of (`tool_returns`, `tool_raises_ToolArgumentError`, `tool_raises_AssertionError`, `tool_raises_Exception`, `audit_raises_IntegrityError`, `audit_raises_OperationalError_on_insert`, `audit_raises_OperationalError_on_commit`, `advisory_lock_unavailable`, `tool_call_cap_exceeded`, `unknown_response_key`).
- `st_cancellation_arrival_time`: enum per spec §8.3.1 expanded list.
- `st_session_state`: synthetic `CompositionState` with version 0–50.

Each strategy should include a `# Maps onto §5.5 row N: ...` comment.

(Full implementation is mechanical; follow Hypothesis idioms. Use `@example` decorators to mechanically guarantee coverage of every enum arm — closes spec QA F-6.)

- [ ] **Step 2: Property test**

Create `tests/property/web/composer/test_compose_loop_invariants.py`:

```python
"""Property test for the bidirectional INV-AUDIT-AHEAD invariant
(spec §8.3). Stateful machine over (LLM emissions, tool executions,
cancellations, redaction policies). Asserts the post-conditions in
§8.3.2 after every trace."""
from __future__ import annotations

from hypothesis import HealthCheck, settings, given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from . import strategies as S


class ComposeLoopMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        # Spin up a fresh in-memory composer service per trace.
        ...

    @rule(tool_call=S.st_tool_call())
    def llm_emits_tool_call(self, tool_call):
        ...

    @rule(injection=S.st_failure_injection_point())
    def inject_failure(self, injection):
        ...

    @rule(arrival=S.st_cancellation_arrival_time())
    def schedule_cancellation(self, arrival):
        ...

    @invariant()
    def forward_direction_holds(self):
        # count(role='assistant' rows for turn) == 1
        # count(role='tool' rows for turn) <= N
        ...

    @invariant()
    def backward_direction_schema_predicate(self):
        # Run the SQL from §8.3.2 — empty result expected.
        ...

    @invariant()
    def otel_counter_post_conditions(self):
        # Per spec §8.3.2: count of failure-mode traces matches counter.
        ...


TestComposeLoop = ComposeLoopMachine.TestCase
TestComposeLoop.settings = settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
```

(Implementation is non-trivial. The full property test campaign should pass before the phase closes.)

- [ ] **Step 3: Run the property test**

```bash
.venv/bin/python -m pytest tests/property/web/composer/test_compose_loop_invariants.py -v
```
Expected: PASS within a reasonable budget. If a counterexample is found, debug and fix the underlying issue rather than reducing the search.

- [ ] **Step 4: Commit**

```bash
git add tests/property/web/composer/
git commit -m "test(property): INV-AUDIT-AHEAD bidirectional invariant property test (composer-progress-persistence phase 3)"
```

---

## Task 14: File OQ-3 follow-up Filigree ticket (integrity hash chain)

- [ ] **Step 1: Create the issue via filigree MCP tool**

- Title: `chat_messages integrity-hash chain follow-up`
- Type: `task`
- Priority: `P3`
- Description (verbatim from spec §10 OQ-3 mechanism sketch):

```
Spec section: docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md §10 OQ-3.

Add per-row integrity binding to chat_messages, composition_states, and
audit_access_log. Mechanism sketch (full text in spec):

1. Add columns: integrity_hash (String(64), NOT NULL), prev_integrity_hash
   (String(64), nullable for the first row of a session).
2. On INSERT, compute:
     prev_hash = SELECT integrity_hash FROM chat_messages
                 WHERE session_id=? ORDER BY sequence_no DESC LIMIT 1
     canonical = canonical_json({...all row fields, including prev_hash...})
     integrity_hash = sha256(canonical).hexdigest()
3. Add verify_chain(session_id) helper.
4. Schedule periodic verifier emitting alerts on mismatch.
5. Optional HMAC variant: hmac.new(deployment_key, canonical, sha256).

This ticket is filed as out-of-scope for the rev-4 composer progress
persistence work because the rev-4 design's auditability posture is
structural (CHECK constraints, provenance discriminator, writer-principal
attribution, access logging) rather than cryptographic.
```

- Labels: `cluster:composer-progress-persistence`, `from-design-spec`, `security-followup`

- [ ] **Step 2: Record the ticket ID**

Cite the assigned `elspeth-XXXXXXXX` ID in the Phase 3 PR description as the resolution to OQ-3.

---

## Task 15: Final Phase 3 CI run

- [ ] **Step 1: Run the full composer + integration suites**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/ tests/integration/web/ tests/integration/pipeline/test_composer_llm_eval_characterization.py tests/property/web/composer/ -v
```
Expected: PASS.

- [ ] **Step 2: Run tier-model, freeze-guard, mypy, ruff**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/ tests/
```
Expected: all green.

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "feat(composer): progress persistence phase 3 — compose-loop persistence + tool-call cap" --body "$(cat <<'EOF'
## Summary

Phase 3 of composer-progress-persistence (spec §11):
- Wires `_compose_loop` through Phase 1's `persist_compose_turn` and Phase 2's redaction primitives
- Plumbs `raw_content=assistant_message.content` into every `persist_compose_turn(...)` call (closes B2-followup-1 from the Phase 1 closure review — the audit-attribution column is non-`NULL` for every compose-turn assistant row, preserving the redaction-attribution invariant end-to-end)
- Adds per-turn tool-call cap (default 16; new `tool_call_cap_exceeded` reason code)
- Surfaces `failed_turn` field on 422/500 error responses
- Exposes `include_tool_rows=true` on the messages endpoint with audit-grade access logging
- Sixteen integration scenarios pass (CL-PP-1 through CL-PP-13)
- Hypothesis property test asserts INV-AUDIT-AHEAD bidirectional invariant

## Spec

`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` revision 4.

## Depends on

Phase 1 PR (data layer + sync primitive) and Phase 2 PR (redaction framework).

## Out of scope (later phases)

- Frontend recovery panel (Phase 4)

## Follow-ups filed

- [OQ-3] elspeth-XXXXXXXX (chat_messages integrity-hash chain — security follow-up)

## Test plan

- [x] Tool-call cap CL-PP-12 (unit)
- [x] CL-PP-1 convergence mid-loop preserves breadcrumbs
- [x] CL-PP-2 plugin crash mid-loop persists rows + raises
- [x] CL-PP-3 wall-clock timeout
- [x] CL-PP-4a/4c DB write failure variants (Tier-1 + unwind)
- [x] CL-PP-5 summarizer-raises end-to-end
- [x] CL-PP-6 cross-session isolation
- [x] CL-PP-7 mid-loop cancellation
- [x] CL-PP-8 duplicate tool_call_id
- [x] CL-PP-9 mixed redaction
- [x] CL-PP-10a/b INSERT-OK-COMMIT-FAIL
- [x] CL-PP-13 unknown-response-key fail-closed
- [x] Audit-grade view emits access-log row
- [x] Hypothesis property test green
- [x] OTel counter post-conditions hold
- [x] tier-model + freeze-guard + mypy + ruff green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Phase 3 Done When

All 15 tasks above are complete. Specifically:

1. ✅ All sixteen CL-PP-* integration scenarios pass.
2. ✅ Hypothesis property test passes with the bidirectional post-conditions.
3. ✅ OTel counter post-conditions hold across the property-test campaign.
4. ✅ `failed_turn` field on error responses; `include_tool_rows=true` query parameter; access-log row emitted.
5. ✅ Per-turn tool-call cap with reason code.
6. ✅ tier-model, freeze-guard, mypy, ruff green.
7. ✅ OQ-3 integrity-hash follow-up filed.
8. ✅ **B2-followup-1 closed:** `raw_content=assistant_message.content` is plumbed into every `persist_compose_turn(...)` call inside `_compose_loop`; the Task 5 happy-path test asserts the column lands non-`NULL` for assistant rows; `grep -n "persist_compose_turn(" src/elspeth/web/composer/service.py` shows no call site missing the kwarg. (Closes the cross-phase coupling gap surfaced during the Phase 1 B2 closure review — the audit-attribution invariant is preserved end-to-end.)

Phase 4 begins after this PR merges. Phase 4 builds the frontend recovery panel that consumes the `failed_turn` and `include_tool_rows=true` surfaces this phase delivered.
