# Composer Progress Persistence — Phase 3: Compose-Loop Persistence + Tool-Call Cap (rev 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Revision history.** Rev 1 (commit `6bcf2f2d`, 2026-05-10) was authored when Phase 2 was planned but unimplemented. Rev 2 (this revision, 2026-05-12) re-baselines against the actual Phase 2 implementation at HEAD `f54ee7e8` — every Phase 2 symbol the plan consumes is now on-disk and gate-green. Rev-2 changes are scoped to: (1) Dependency posture section truthful re: Phase 2 shipped; (2) Ground rules drops the "import-time RED against Phase 2 symbols" framing; (3) Task 5 Step 1 RED expectation re-framed (no more `ImportError`); (4) Task 16 final step removes the unconditional `gh pr create` and replaces it with operator-surface — per Phase 2 rev-5 BLOCKER B4 closure pattern, per project memory `feedback_default_to_worktree` (worktree-default policy 2026-05-11) and per the operator's standing instruction to defer PR-open. The task structure and TDD steps are otherwise unchanged. Appendix A (new) carries Phase-2-as-shipped traceability — F2 / F3 / AST-fingerprint observation references — that Phase 3 implementers will encounter at mid-file edit sites.

**Goal:** Wire the live compose loop through the Phase 1 persistence primitive so a failed compose turn lands its assistant row, tool rows, and composition state atomically; enforce the per-turn tool-call cap; surface `failed_turn` + `tool_responses_persisted` on 422/500 error bodies; and expose `include_tool_rows=true` on `GET /api/sessions/{sid}/messages` with audit-grade access logging.

**Architecture:** Compose-loop body in `src/elspeth/web/composer/service.py:_compose_loop` is rewritten to the §5.2.1 single-sync-block shape — gather tool outcomes async, redact via the Phase 2 manifest walker, dispatch ONE `persist_compose_turn_async` per turn. Phase 1 owns the schema, the sync primitive, and the advisory lock; Phase 2 owns the redaction manifest; this phase owns the loop + route shape + transcript endpoint. The old `BufferingRecorder.add_message` drain path inside `_compose_loop` is **deleted in this PR** (no-legacy-code policy).

**Tech Stack:** Python 3.13, SQLAlchemy 2.x (sync `Engine`), FastAPI, structlog, OpenTelemetry, Hypothesis (property tests), pytest, testcontainer PostgreSQL on the Docker-enabled CI lane.

**Spec sections:**
§5.2 (insertion sites — loop shape, sync write function, what this design eliminates, cancellation semantics, atomicity grain),
§5.4 (`partial_state` redaction symmetry),
§5.5 (failure-mode interaction table — twelve rows),
§5.7.5 (`MANIFEST` lookup integration),
§6.1 (`_handle_*` `failed_turn` field, read-consistency note),
§6.2 (transcript endpoint + `include_tool_rows`),
§6.3 (audit-grade access logging + `audit_access_log` writes),
§8.2 (CL-PP-1..13 integration tests),
§8.3 (property test — `INV-AUDIT-AHEAD` bidirectional),
§8.5.1 (testcontainer CI lane),
§11 Phase 3 scope + done-when.

**Spec revision:** This plan consumes **revision 5** of the spec (manifest-keyed redaction; `AuditOutcome` two-field shape; `AuditIntegrityError`-raising Tier-1 disposition). Cite the spec body, not the rev-4 §11 supersession block.

---

## Dependency posture (rev-2 update — Phase 2 is now shipped)

**This plan executes on `feat/composer-progress-persistence-1a` at HEAD `f54ee7e8` or later (the umbrella branch per memory `project_adr010_umbrella_branch`).** All upstream Phase 1A/1B/1C and Phase 2 deliverables are landed.

- Phase 1 (1A/1B/1C) is **shipped** on this branch. The compose loop calls `persist_compose_turn_async`, `RedactedToolRow`, `StatePayload`, `CompositionStateData`, `AuditOutcome`, `StaleComposeStateError`, and `AuditIntegrityError` — all already importable from `src/elspeth/web/sessions/protocol.py`, `_persist_payload.py`, and `elspeth.contracts.errors`. `_ToolOutcome` exists at `web/sessions/_persist_payload.py:89` (frozen, `freeze_fields(self, "call", "response")`); Phase 3 imports it, does NOT redefine. The `audit_access_log_table` is defined INERT at `web/sessions/models.py:485`; Task 11 wires the writer.
- Phase 2 (manifest-keyed dispatch + `Sensitive[T]` promotion wave + adequacy guard + F1–F6 follow-ups) is **shipped** as of HEAD `f54ee7e8`. The Phase 3 plan's red tests against Phase 2 symbols now fail with `AssertionError` (wiring not yet present), not `ImportError`. Specifically, every Phase 2 surface this plan consumes is on disk:
  - `MANIFEST: Mapping[str, ToolRedaction]` at `web/composer/redaction.py:2352` (38 entries — 10 type-driven + 28 declarative per `project_phase2_implementation_complete`).
  - `redact_tool_call_arguments(tool_name, decoded_args, telemetry=...)` at `web/composer/redaction.py:1658`.
  - `redact_tool_call_response(tool_name, response, telemetry=...)` at `web/composer/redaction.py:2635`.
  - `class RedactionTelemetry(Protocol)` at `web/composer/redaction_telemetry.py:32`; `NoopRedactionTelemetry` at `:42`; `OtelRedactionTelemetry` at `:60`.
  - `_arg_error_payload(exc, tool_name) -> Mapping[str, Any]` at `web/composer/service.py:3705` (F2 — module-tail helper preserving AST fingerprints; populates `validation_errors` field on ARG_ERROR `result_canonical` when `exc.__cause__` is a `pydantic.ValidationError`).
  - `canonicalize_pydantic_cause(exc)` at `web/composer/audit.py:826` (F2 — leak-safe `loc`/`msg`/`type` canonicalisation, `input`/`url`/`ctx` stripped).

  Phase 3 consumes these symbols verbatim; the walker dispatches through `MANIFEST[tool_name]` (spec §5.7.5); `_arg_error_payload` is the canonical ARG_ERROR factory invoked by both `dispatch_with_audit` and the `except ToolArgumentError` arm in `_compose_loop`.
- Phase 1 also shipped CL-PP-11 (concurrent multi-session writes against testcontainer Postgres — commit `eca88974`). Phase 3 does not re-author that test; it cites the existing test as the CL-PP-11 deliverable in the §11 done-when checklist.
- **Sequencing for Phase 3 execution:** the implementer starts at HEAD `f54ee7e8` (or later) on `feat/composer-progress-persistence-1a`. No upstream phase is pending. Reds in Phase 3 tasks fail because the loop body is unwired, not because imports are unresolved.

---

## File Structure

### Files to modify (existing)

- `src/elspeth/web/composer/service.py` — `_compose_loop` body rewritten to the §5.2.1 shape; per-turn tool-call cap; legacy `BufferingRecorder.add_message` drain inside the loop deleted; tool-result accumulation switched to `_ToolOutcome`.
- `src/elspeth/web/composer/protocol.py` — `ComposerConvergenceError.capture` accepts the new `tool_call_cap_exceeded` reason code; `ComposerProgressEvent` reason-code enum gains `tool_call_cap_exceeded`.
- `src/elspeth/web/sessions/routes.py` — `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure` each add the `failed_turn` field; `GET /api/sessions/{sid}/messages` gains `include_tool_rows: bool = False` and emits an `audit_access_log` row when `include_tool_rows=true`.
- `src/elspeth/web/sessions/service.py` — adds `count_tool_responses_for_assistant(assistant_message_id) -> int` and `record_audit_grade_view(session_id, requesting_principal, request_path, query_args, ip_address) -> None`.
- `src/elspeth/web/sessions/protocol.py` — `SessionServiceProtocol` gains `count_tool_responses_for_assistant_async` and `record_audit_grade_view_async`.
- `src/elspeth/web/sessions/telemetry.py` — `_SessionsTelemetry` extended with `tool_call_cap_exceeded_total` and `audit_grade_view_total`; `build_sessions_telemetry` wires both in both fake and real branches (the telemetry.py:126–132 docstring already documents this Phase-3 extension).
- `src/elspeth/web/sessions/schemas.py` — response model for the messages endpoint gains `tool_call_id`, `parent_assistant_id`, `sequence_no` fields (Phase 1 already added the columns; Phase 3 exposes them on the API surface).
- `docs/superpowers/plans/2026-04-30-composer-progress-persistence-overview.md` — Task 14 below updates the overview to rev-5 wording (currently still says "revision 4" / `Sensitive[T]` at lines 5, 9, 20, 27, 29).

### Files to create (new)

- `tests/unit/web/composer/test_compose_loop_tool_call_cap.py` — Step 0 unit tests.
- `tests/unit/web/composer/test_compose_loop_persistence.py` — Step 1/2/3 unit tests against in-memory SQLite via `create_session_engine` + `initialize_session_schema`.
- `tests/unit/web/composer/test_audit_failure_primacy.py` — `AuditIntegrityError` propagation under tool-success/tool-fail interaction.
- `tests/unit/web/sessions/test_count_tool_responses_for_assistant.py` — read helper.
- `tests/unit/web/sessions/test_record_audit_grade_view.py` — access-log write helper.
- `tests/unit/web/sessions/test_messages_route_include_tool_rows.py` — endpoint behaviour + access-log emission.
- `tests/integration/web/test_inv_audit_ahead_backward.py` — schema-level backward-direction post-condition (spec §4.1.2 / §5.3 / §8.3.2).
- `tests/integration/web/test_compose_loop_failed_turn_field.py` — `failed_turn` response shape across all three `_handle_*` helpers.
- `tests/property/web/composer/test_compose_loop_invariants.py` — Hypothesis property test (§8.3).
- `tests/property/web/composer/strategies.py` — strategy contracts (§8.3.1) if not already in place.

Existing characterization tests in `tests/integration/pipeline/test_composer_llm_eval_characterization.py` are **extended in place** to author/refresh CL-PP-1..10b, 12, 13. CL-PP-11 (commit `eca88974`) is unchanged.

### Files NOT touched in Phase 3

- `src/elspeth/web/composer/redaction.py` — Phase 2 owns this. Phase 3 imports the walker entry points.
- `src/elspeth/web/composer/tools.py` — Phase 2's promotion wave touches it. Phase 3 does not.
- `src/elspeth/web/sessions/models.py` — Phase 1 owns schema. The `audit_access_log` table and the new `chat_messages` columns are already in place.
- `src/elspeth/web/frontend/**` — Phase 4 owns the recovery panel.

---

## Ground rules

- **TDD throughout.** Every task ends in a failing test before any implementation lands. The reds are wiring-level (`AssertionError` / `AttributeError`), not import-level — Phase 2 symbols resolve at HEAD `f54ee7e8` (see Dependency posture; the rev-1 "import-time RED against Phase 2 symbols" framing is superseded). Do not stub Phase 2 surfaces; they exist.
- **Real databases in tests.** `create_session_engine(..., poolclass=StaticPool)` + `initialize_session_schema()` for SQLite (per spec §8.6); testcontainer PostgreSQL for CL-PP-11 (already wired) and any new PostgreSQL-only assertion. Bare `metadata.create_all()` is banned. Mocking `persist_compose_turn` or any `_insert_*` helper is banned.
- **No defensive programming.** Tier-1 audit data crashes on anomaly. `AuditIntegrityError` is the canonical signal; it is in `TIER_1_ERRORS` and must not be caught by `except Exception`. The compose loop's `except Exception as tool_exc:` branch (the plugin-bug surface) **must** re-raise `AuditIntegrityError` unmodified — capture only via the existing `service.py:942–980` pattern.
- **No legacy code.** The current `_compose_loop` accumulates tool results into `BufferingRecorder` and drains via `_persist_tool_invocations` after the loop ends. The §5.2.1 shape supersedes that. Delete the legacy path in the same PR; do not leave both wired behind a feature flag.
- **Audit primacy.** Logging is permitted only on the unwind-audit-failed path (`AuditOutcome.unwind_audit_failed=True`) where the audit-system is the failing surface; everywhere else, telemetry counters carry the operational signal.
- **`from exc` chaining preserved.** Existing `ComposerPluginCrashError.capture` already sets `__cause__`; the rewrite must keep that chain intact when re-raising after the audit write.
- **Frequent commits.** Each task ends with a commit; tasks with subtasks commit per subtask.

---

## Task 1: Verify the `_ToolOutcome` payload dataclass

The compose loop's per-iteration outcome record already exists at `src/elspeth/web/sessions/_persist_payload.py:90` (`_ToolOutcome`). Phase 3 imports it; it does not author or extend it unless a field is missing.

**Files:**
- Read-only: `src/elspeth/web/sessions/_persist_payload.py`

- [ ] **Step 1: Confirm the field list matches §5.2.1's loop body.**

Required fields per spec §5.2.1: `call: Any`, `response: Any`, `error_class: str | None`, `error_message: str | None`, `pre_version: int`, `post_version: int`. Verify by reading `_persist_payload.py:113–124` directly. The dataclass is `frozen=True, slots=True` with a `freeze_fields(self, "call", "response")` post-init guard.

If every field matches, this task is a verification-only no-op and is closed by commenting in the PR description: "Task 1 verified: `_ToolOutcome` fields match spec §5.2.1; no Phase 3 changes required to `_persist_payload.py`." No commit.

If a field is missing or has the wrong type, treat this as a Phase 1 hygiene defect: file a Filigree issue with reproducer steps, link from the PR, and add the missing field on this branch with a per-field test in `tests/unit/web/sessions/test_persist_payload.py` (extending the existing file). Do not add fields that §5.2.1 does not call for.

- [ ] **Step 2: Commit (only if Step 1 added a field).**

Commit message: `fix(sessions): add missing _ToolOutcome.<field> per spec §5.2.1 (composer-progress-persistence phase 3)`.

---

## Task 2: Per-turn tool-call cap configuration + reason code

The §1.4 NFR caps tool calls per assistant turn at 16 (default), env-tunable. The cap raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")` **before** any tool execution. Step 0 of the §5.2.1 loop body depends on this.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (ComposerServiceImpl constructor: accept and store `max_tool_calls_per_turn`)
- Modify: `src/elspeth/web/composer/protocol.py` (`ComposerProgressEvent` reason-code enum gains `tool_call_cap_exceeded`; `ComposerConvergenceError.capture` accepts and propagates the new reason)
- Modify: `src/elspeth/web/sessions/telemetry.py` (`_SessionsTelemetry` gains `tool_call_cap_exceeded_total`; `build_sessions_telemetry` wires it in both fake and real branches)
- Modify: `src/elspeth/config.py` and the matching Settings→Runtime contract (see `config-contracts-guide` skill) for the env-tunable `MAX_TOOL_CALLS_PER_TURN`
- Create: `tests/unit/web/composer/test_compose_loop_tool_call_cap.py`

- [ ] **Step 1: Write the failing red test.**

In `tests/unit/web/composer/test_compose_loop_tool_call_cap.py`:

```python
"""Tests for the per-turn tool-call cap (spec §1.4 NFR / §5.2.1 Step 0).

Drives the compose loop with an LLM that emits more tool calls in one
assistant turn than the cap allows. The loop must raise
ComposerConvergenceError(reason="tool_call_cap_exceeded") BEFORE any
tool execution, and increment composer.tool_call_cap_exceeded_total.
"""
from __future__ import annotations

import pytest

from elspeth.web.composer.protocol import (
    ComposerConvergenceError,
    ComposerProgressEvent,
)
from elspeth.web.sessions.telemetry import build_sessions_telemetry, observed_value


def test_cap_exceeded_raises_before_any_tool_execution(
    fake_composer_service, fake_llm_emitting_n_tool_calls
):
    # Cap is 16 by default; LLM emits 17.
    fake_llm = fake_llm_emitting_n_tool_calls(n=17)
    fake_composer_service._max_tool_calls_per_turn = 16
    fake_composer_service._llm = fake_llm

    with pytest.raises(ComposerConvergenceError) as excinfo:
        fake_composer_service._run_one_turn_for_test()

    # The reason code is the new tool_call_cap_exceeded variant.
    assert excinfo.value.reason == ComposerProgressEvent.Reason.tool_call_cap_exceeded
    assert excinfo.value.evidence["observed"] == 17
    assert excinfo.value.evidence["cap"] == 16

    # No tool execution attempted — execute_tool must not have been
    # called at all. fake_llm tracks calls into execute_tool via a
    # spy installed in the conftest fixture.
    assert fake_llm.execute_tool_invocations == 0


def test_cap_exceeded_increments_counter(
    fake_composer_service, fake_llm_emitting_n_tool_calls
):
    telemetry = build_sessions_telemetry()  # fake counters
    fake_composer_service._telemetry = telemetry
    fake_composer_service._max_tool_calls_per_turn = 16
    fake_composer_service._llm = fake_llm_emitting_n_tool_calls(n=17)

    with pytest.raises(ComposerConvergenceError):
        fake_composer_service._run_one_turn_for_test()

    assert observed_value(telemetry.tool_call_cap_exceeded_total) == 1


def test_cap_not_exceeded_does_not_increment(
    fake_composer_service, fake_llm_emitting_n_tool_calls
):
    telemetry = build_sessions_telemetry()
    fake_composer_service._telemetry = telemetry
    fake_composer_service._max_tool_calls_per_turn = 16
    fake_composer_service._llm = fake_llm_emitting_n_tool_calls(n=16)

    fake_composer_service._run_one_turn_for_test()  # no raise

    assert observed_value(telemetry.tool_call_cap_exceeded_total) == 0
```

Run the test:

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_tool_call_cap.py -v
```

Expected: FAIL — `ComposerProgressEvent.Reason.tool_call_cap_exceeded` does not exist; `_max_tool_calls_per_turn` is not an attribute on the service; `tool_call_cap_exceeded_total` is not a field on `_SessionsTelemetry`.

- [ ] **Step 2: Add the reason code + telemetry counter.**

In `src/elspeth/web/composer/protocol.py`, extend `ComposerProgressEvent.Reason` (or its current `Literal[...]` equivalent — verify against the file's current shape) with `"tool_call_cap_exceeded"`. Update `ComposerConvergenceError.capture` to accept the reason and `evidence: Mapping[str, Any]` extension carrying `observed` and `cap`.

In `src/elspeth/web/sessions/telemetry.py:120` (`_SessionsTelemetry`), append the field after `tool_row_integrity_violation_total`:

```python
tool_call_cap_exceeded_total: _Counter
```

In `build_sessions_telemetry` (both `meter is None` and real branches), add the matching entry:

```python
tool_call_cap_exceeded_total=_FakeCounter(),
# ...
tool_call_cap_exceeded_total=meter.create_counter(
    "composer.tool_call_cap_exceeded_total"
),
```

Update the docstring comment at telemetry.py:126–132 to drop the "Phase 3 (compose loop + audit-grade view) adds" forward-looking line, since this PR is delivering it.

- [ ] **Step 3: Wire `_max_tool_calls_per_turn` into the composer service.**

Add `max_tool_calls_per_turn: int = 16` to `ComposerServiceImpl.__init__`, store as `self._max_tool_calls_per_turn`. Source the value from runtime config via the existing `from_settings(...)` mapping (see `config-contracts-guide` skill): add `MAX_TOOL_CALLS_PER_TURN: int = 16` to the Composer settings dataclass, route it through the contract, and confirm the `check_contracts` script passes:

```bash
.venv/bin/python -m scripts.check_contracts
```

- [ ] **Step 4: Re-run the test to verify GREEN.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_tool_call_cap.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add -p src/elspeth/web/composer/protocol.py \
            src/elspeth/web/composer/service.py \
            src/elspeth/web/sessions/telemetry.py \
            src/elspeth/config.py \
            src/elspeth/contracts/config/composer.py \
            tests/unit/web/composer/test_compose_loop_tool_call_cap.py
git commit -m "feat(composer): per-turn tool-call cap + tool_call_cap_exceeded reason code (composer-progress-persistence phase 3)"
```

---

## Task 3: Rewrite `_compose_loop` body — Step 0 (cap enforcement)

Implement Step 0 of the §5.2.1 loop body inside `_compose_loop`. The check runs **before** the for-loop over `assistant_message.tool_calls` opens; no tool execution happens for the over-cap turn.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` method body, after the LLM call returns)

- [ ] **Step 1: Read the current `_compose_loop` body.**

Read `web/composer/service.py:1724–1900` (approximate range — the `async def _compose_loop` definition). Identify where `assistant_message` becomes available after the LLM call and where the existing per-tool for-loop begins. Step 0 sits between them.

- [ ] **Step 2: Insert the Step 0 check.**

Immediately after `assistant_message` is bound and before any iteration over `assistant_message.tool_calls` begins, insert:

```python
# Step 0 — enforce the per-turn tool-call cap (spec §1.4 NFR / §5.2.1).
# RSK-13: bounds the cost of a single turn that a misbehaving LLM
# might inflate. Raises BEFORE any tool execution; no DB writes for
# the over-cap turn; the existing _handle_convergence_error path
# downstream emits partial_state captured pre-cap.
if len(assistant_message.tool_calls) > self._max_tool_calls_per_turn:
    self._telemetry.tool_call_cap_exceeded_total.add(1)
    raise ComposerConvergenceError.capture(
        state,
        reason=ComposerProgressEvent.Reason.tool_call_cap_exceeded,
        evidence={
            "observed": len(assistant_message.tool_calls),
            "cap": self._max_tool_calls_per_turn,
        },
        initial_version=initial_version,
    )
```

Telemetry increments BEFORE the raise so the counter is incremented even if the exception path is later caught and re-raised differently by `_handle_convergence_error`.

- [ ] **Step 3: Re-run Task 2's tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_tool_call_cap.py -v
```

Expected: PASS (this fully closes the cap-exceeded surface).

- [ ] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 0 — enforce per-turn tool-call cap (composer-progress-persistence phase 3)"
```

---

## Task 4: Rewrite `_compose_loop` body — Step 1 (tool-execution accumulation)

Convert the existing per-tool-call execution into the §5.2.1 Step 1 shape: accumulate `_ToolOutcome` records, distinguish ToolArgumentError / interpreter-invariant / plugin-bug arms, capture `plugin_crash` without re-raising, and break out of the for-loop on a plugin crash so Step 2 still runs.

The loop body must also track two load-bearing variables for Step 2:

1. **`current_state_id`** — the state id observed BEFORE the LLM call. This is the `expected_current_state_id` argument to `persist_compose_turn_async` (§5.2.1 line 1498). Capture it as a local at the top of the turn (before the LLM call) so it survives the `await` and is passed into Step 2 unchanged.

2. **`pre_state_id_for(outcome)`** — for each `_ToolOutcome`, the state id observed just before that tool call ran. Used to populate `StatePayload.derived_from_state_id` in Step 2. Maintain as a per-iteration local; rebind after each tool call by reading the current latest-state id (via a new SessionsService read helper or by tracking the last-inserted state id inside the loop's own bookkeeping — preferred, no extra DB roundtrip). The compose loop owns this bookkeeping; do not surface a SessionsService helper for it.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body)

- [ ] **Step 1: Write the failing red test.**

Create `tests/unit/web/composer/test_compose_loop_persistence.py`:

```python
"""Compose-loop Step 1/2/3 unit tests (spec §5.2.1).

These tests drive the loop against an in-memory SQLite SessionsService
built via create_session_engine + initialize_session_schema (per spec
§8.6 — bare metadata.create_all is banned for this surface). They assert
the §5.5 failure-mode rows 1, 2, and 4 individually.
"""
from __future__ import annotations

import pytest

from elspeth.web.sessions.engine import create_session_engine, initialize_session_schema
from sqlalchemy.pool import StaticPool


@pytest.fixture
def sessions_service():
    engine = create_session_engine("sqlite://", poolclass=StaticPool)
    initialize_session_schema(engine)
    # ... (existing fixture pattern from tests/unit/web/sessions/test_persist_compose_turn.py)


def test_step1_three_tools_all_succeed_accumulates_three_outcomes(
    composer_service_with_real_sessions, fake_llm_three_tool_calls
):
    """Step 1 happy path — three successful tools produce three outcomes
    with response set, error_class None, and post_version tracking."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_three_tool_calls
    )
    outcomes = result.tool_outcomes_for_assertion  # test hook
    assert len(outcomes) == 3
    assert all(o.error_class is None for o in outcomes)
    assert all(o.response is not None for o in outcomes)
    # post_version monotonic non-decreasing across outcomes:
    assert outcomes[0].post_version <= outcomes[1].post_version <= outcomes[2].post_version


def test_step1_tool_argument_error_continues_loop(
    composer_service_with_real_sessions, fake_llm_tool_argument_error_on_second
):
    """ToolArgumentError on call 2 of 3 — outcome 2 has
    error_class='ToolArgumentError'; loop continues to call 3."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_tool_argument_error_on_second
    )
    outcomes = result.tool_outcomes_for_assertion
    assert len(outcomes) == 3
    assert outcomes[0].error_class is None
    assert outcomes[1].error_class == "ToolArgumentError"
    assert outcomes[2].error_class is None


def test_step1_assertion_error_reraises_before_persist(
    composer_service_with_real_sessions, fake_llm_assertion_error_on_second
):
    """AssertionError on call 2 of 3 — re-raised BEFORE Step 2 runs.
    No DB writes happen for the failed turn; route-helper partial-state
    persist covers the snapshot."""
    with pytest.raises(AssertionError):
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_assertion_error_on_second
        )
    # No chat_messages row for this turn — assistant or tool:
    with sessions_service._engine.connect() as conn:
        rows = conn.execute(...).fetchall()
        assert rows == []


def test_step1_plugin_bug_captures_crash_breaks_loop(
    composer_service_with_real_sessions, fake_llm_runtime_error_on_second
):
    """RuntimeError on call 2 of 3 — outcome 2 has error_class='RuntimeError';
    loop breaks (no call 3); plugin_crash is captured for raise-after-Step-2."""
    with pytest.raises(ComposerPluginCrashError) as excinfo:
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_runtime_error_on_second
        )
    # The audit write DID happen — assistant + 2 tool rows visible:
    # (one normal outcome 1, one error outcome 2; outcome 3 never executed)
    # The captured exception preserves the original cause chain.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RuntimeError)
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py::test_step1_three_tools_all_succeed_accumulates_three_outcomes -v
```

Expected: FAIL — the loop currently drains to `BufferingRecorder`, not `_ToolOutcome[]`. The test hook `tool_outcomes_for_assertion` does not exist.

- [ ] **Step 2: Refactor the per-tool for-loop body to accumulate `_ToolOutcome`.**

In `_compose_loop`, replace the current per-tool record/drain pattern with the §5.2.1 Step 1 shape:

```python
# Step 1 — execute every tool call in async land, accumulating
# _ToolOutcome records. No audit work writes yet; cancellation here
# is safe (the LLM response is in memory, the DB is unchanged).
tool_outcomes: list[_ToolOutcome] = []
plugin_crash: ComposerPluginCrashError | None = None
pre_state_id = current_state_id  # captured before LLM call; load-bearing for Step 2

for tool_call in assistant_message.tool_calls:
    pre_version = state.version
    try:
        response = await execute_tool(tool_call, state)
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=response,
            error_class=None,
            error_message=None,
            pre_version=pre_version,
            post_version=state.version,
        ))
        # If the tool advanced state, the previous state row's id is
        # the new pre_state_id for the next iteration. State id
        # bookkeeping is loop-local — no SessionsService roundtrip
        # because persist_compose_turn re-reads MAX(version) under
        # the held write lock anyway (spec §5.7.1).
        if state.version > pre_version:
            # The new state id will be allocated inside
            # persist_compose_turn (Step 2); for now retain the
            # pre-call state id as the predecessor for this outcome's
            # StatePayload.derived_from_state_id. See Step 2 below.
            pass
    except ToolArgumentError as exc:
        # Tier-3 boundary signal. The loop continues; the LLM gets
        # the error tool row as feedback. Existing contract preserved
        # from service.py:867 / protocol.py:299-303.
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=None,
            error_class="ToolArgumentError",
            error_message=str(exc),
            pre_version=pre_version,
            post_version=state.version,
        ))
    except (AssertionError, MemoryError, RecursionError, SystemError):
        # Interpreter-level invariant violations. Not recoverable, not
        # wrappable. Re-raise BEFORE Step 2 runs; route-helper
        # partial-state persist runs from _handle_*. service.py:907.
        raise
    except AuditIntegrityError:
        # Tier-1 invariant raised by execute_tool or downstream.
        # Re-raise unmodified — registered in TIER_1_ERRORS so this
        # except Exception block below CANNOT swallow it (the bare
        # AuditIntegrityError clause runs first).
        raise
    except Exception as tool_exc:
        # Plugin-bug surface. service.py:942–980 pattern: wrap with
        # ComposerPluginCrashError.capture, break out, raise AFTER
        # the Step-2 audit write. The audit row for the crashing tool
        # is committed in the same transaction as the surviving tool
        # rows; the crash signal propagates only after audit is durable.
        tool_outcomes.append(_ToolOutcome(
            call=tool_call,
            response=None,
            error_class=type(tool_exc).__name__,
            error_message=str(tool_exc),
            pre_version=pre_version,
            post_version=state.version,
        ))
        plugin_crash = ComposerPluginCrashError.capture(
            tool_exc,
            state=state,
            initial_version=initial_version,
        )
        break
```

**Critical:** the `except AuditIntegrityError:` clause is positioned BEFORE the catch-all `except Exception as tool_exc:` arm. `AuditIntegrityError` is decorated with `@tier_1_error` (registered in `TIER_1_ERRORS`), but `except Exception` will still catch it without the dedicated clause; the explicit re-raise is the offensive-programming guard. CLAUDE.md no-defensive-programming: do not wrap this in `except (AuditIntegrityError,) as exc: raise`. The bare `raise` is correct.

Delete the now-orphaned `BufferingRecorder.add_message` calls within the loop body. The recorder remains in scope for non-tool events (LLM call telemetry, request-level audit), but `add_message` is no longer the path for per-tool persistence.

- [ ] **Step 3: Re-run the Step 1 tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v -k "step1"
```

Expected: PASS for `test_step1_three_tools_all_succeed_accumulates_three_outcomes`, `test_step1_tool_argument_error_continues_loop`, and `test_step1_assertion_error_reraises_before_persist`. The plugin-bug test (`test_step1_plugin_bug_captures_crash_breaks_loop`) still fails because Step 2 isn't wired — it expects the audit row to exist, which only happens after Step 2 lands.

- [ ] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 1 — _ToolOutcome accumulation + crash capture (composer-progress-persistence phase 3)"
```

---

## Task 5: `_compose_loop` Step 2 — async-side redaction via the Phase 2 manifest

Build the redacted assistant `tool_calls` tuple and the `RedactedToolRow` tuple in async land, calling Phase 2's `redact_tool_call_arguments` and `redact_tool_call_response`. The walker takes the already-decoded arguments dict and a `RedactionTelemetry` instance; the manifest is module-global.

**Dependency:** Phase 2 must be merged. Until then this task's tests fail at import time on `redact_tool_call_arguments` / `redact_tool_call_response`. That is the expected red.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body, after the Task 4 Step 1 block)

- [ ] **Step 1: Write the failing red test.**

Append to `tests/unit/web/composer/test_compose_loop_persistence.py`:

```python
from elspeth.web.composer.redaction import (
    MANIFEST,
    RedactionTelemetry,
    redact_tool_call_arguments,
    redact_tool_call_response,
)


def test_step2_redacts_via_manifest_walker(
    composer_service_with_real_sessions, fake_llm_with_sensitive_tool_call
):
    """Step 2 — redact_tool_call_arguments produces the canonical
    redacted form before persist_compose_turn_async fires. The persisted
    chat_messages.tool_calls JSON is byte-identical to the walker output."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_with_sensitive_tool_call
    )
    # The persisted assistant row's tool_calls JSON matches the walker.
    expected = tuple(
        redact_tool_call_arguments(
            tc.function.name,
            decoded_args[tc.id],
            telemetry=composer_service_with_real_sessions._redaction_telemetry,
        )
        for tc in result.assistant_message.tool_calls
    )
    persisted = result.persisted_assistant_tool_calls
    assert persisted == expected


def test_step2_redacts_response_with_summarizer(
    composer_service_with_real_sessions, fake_llm_summarizer_active
):
    """The redacted tool-row content matches redact_tool_call_response."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_summarizer_active
    )
    expected_content = redact_tool_call_response(
        tool_name=result.tool_outcomes[0].call.function.name,
        response=result.tool_outcomes[0].response,
        telemetry=composer_service_with_real_sessions._redaction_telemetry,
    )
    assert result.persisted_tool_row_content[0] == expected_content
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py::test_step2_redacts_via_manifest_walker -v
```

Expected (rev-2 update — Phase 2 is shipped at HEAD `f54ee7e8`): FAIL with `AssertionError` on `assert persisted == expected` (or similar wiring assertion). Phase 2 imports resolve cleanly; the red is the loop body not yet calling the walker. (Rev-1 said "ImportError until Phase 2 merges"; that expectation is superseded.)

- [ ] **Step 2: Wire the redaction step in `_compose_loop`.**

After the Task 4 Step 1 for-loop, before the (Task 6) dispatch:

```python
# Step 2a — redact in async land. All redaction is pure / non-blocking;
# building the redacted payload here keeps the sync worker's wall-clock
# window narrow. Phase 2 owns the walker; Phase 3 calls it via tool name
# (no lookup_tool_class helper — rev-5 §5.7.5 retired that pattern).
redacted_assistant_tool_calls = tuple(
    redact_tool_call_arguments(
        tc.function.name,
        decoded_args[tc.id],  # M3: arguments parsed once at LLM-call time
        telemetry=self._redaction_telemetry,
    )
    for tc in assistant_message.tool_calls
)

redacted_tool_rows = tuple(
    RedactedToolRow(
        tool_call_id=outcome.call.id,
        content=_serialize_response_via_walker(
            outcome, telemetry=self._redaction_telemetry
        ),
        composition_state_payload=(
            StatePayload(
                data=CompositionStateData(
                    source=state.source,
                    nodes=state.nodes,
                    edges=state.edges,
                    outputs=state.outputs,
                    metadata_=state.metadata_,
                    is_valid=state.is_valid,
                    validation_errors=state.validation_errors,
                    composer_meta=state.composer_meta,
                ),
                derived_from_state_id=pre_state_id_for(outcome, tool_outcomes),
            )
            if outcome.post_version > outcome.pre_version
            else None
        ),
    )
    for outcome in tool_outcomes
)
```

Where `_serialize_response_via_walker` is a private helper on `ComposerServiceImpl` that calls `redact_tool_call_response` on a success outcome and produces the canonical error-tool-row JSON on a failure outcome:

```python
def _serialize_response_via_walker(
    self,
    outcome: _ToolOutcome,
    *,
    telemetry: RedactionTelemetry,
) -> str:
    if outcome.error_class is None:
        redacted = redact_tool_call_response(
            tool_name=outcome.call.function.name,
            response=outcome.response,
            telemetry=telemetry,
        )
        return canonical_json(redacted)  # existing canonical helper
    # Error outcome — the LLM sees a structured error response; the
    # persisted content is the same shape. No redaction needed for the
    # error class/message (no sensitive payload).
    return canonical_json({
        "error_class": outcome.error_class,
        "error_message": outcome.error_message,
    })
```

And `pre_state_id_for(outcome, all_outcomes)` is a small module-level helper (or a closure inside `_compose_loop`):

```python
def pre_state_id_for(
    outcome: _ToolOutcome,
    all_outcomes: Sequence[_ToolOutcome],
) -> str | None:
    """Return the predecessor state id for the given outcome's
    StatePayload.derived_from_state_id.

    Phase 1 allocates new state ids inside persist_compose_turn under
    the held session write lock, so this helper cannot return the new
    state id — it returns the LAST KNOWN COMMITTED state id at the
    moment this outcome's tool call BEGAN. That is the pre-LLM state
    id for the first state-advancing outcome in the turn, and for
    subsequent state-advancing outcomes the loop must track ids as
    they are allocated. For Phase 3 the simplest correct behaviour is
    to set derived_from_state_id=None for every state-advancing
    outcome and let persist_compose_turn re-derive the lineage from
    composition_states.version ordering — Phase 1's
    _insert_composition_state already does this when
    derived_from_state_id is None. A tighter pre-state-id surface is
    a forward-looking refinement filed under OQ-3-adjacent.
    """
    return None  # Phase 1 _insert_composition_state handles the lineage.
```

`pre_state_id_for` is intentionally trivial in Phase 3 because Phase 1 already encodes the version-ordering lineage in `composition_states`; carrying an explicit predecessor id through the loop would duplicate that information. The helper exists so a future tightening (per-tool-row explicit lineage) has a single landing site. Document this decision in a load-bearing comment on the helper.

- [ ] **Step 3: Re-run the Step 2 tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v -k "step2"
```

Expected: PASS. (Phase 2 is shipped at HEAD `f54ee7e8`; rev-1's "PASS once Phase 2 is in" is superseded — see Dependency posture.)

- [ ] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 2 — async-side manifest redaction (composer-progress-persistence phase 3)"
```

---

## Task 6: `_compose_loop` Step 2 — single sync dispatch via `persist_compose_turn_async`

Replace the existing `BufferingRecorder.add_message`-driven persistence with a single `await self.sessions_service.persist_compose_turn_async(...)` call per turn. The legacy drain path inside `_compose_loop` is deleted in this task.

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body, after Task 5; **delete** the existing post-loop drain via `BufferingRecorder`)

- [ ] **Step 1: Write the failing red test.**

Append to `tests/unit/web/composer/test_compose_loop_persistence.py`:

```python
def test_step2_dispatches_one_persist_compose_turn_async_per_turn(
    composer_service_with_real_sessions, fake_llm_two_tool_calls,
    sqlalchemy_event_listener,
):
    """Exactly one BEGIN/COMMIT pair per turn — verified via SQLAlchemy
    event listener counting transaction lifecycle events."""
    composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_two_tool_calls
    )
    assert sqlalchemy_event_listener.begin_count == 1
    assert sqlalchemy_event_listener.commit_count == 1
    assert sqlalchemy_event_listener.rollback_count == 0


def test_step2_passes_raw_content_for_B2_attribution(
    composer_service_with_real_sessions, fake_llm_preflight_rewrites_content,
):
    """The pre-redaction LLM output is passed as raw_content; the
    redacted form is passed as assistant_content. B2 audit-attribution
    is preserved when runtime preflight rewrites the visible content."""
    result = composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_preflight_rewrites_content
    )
    persisted_assistant = result.persisted_assistant_row
    assert persisted_assistant.raw_content == fake_llm_preflight_rewrites_content.original_text
    assert persisted_assistant.content != persisted_assistant.raw_content


def test_step2_does_not_call_legacy_add_message_inside_loop(
    composer_service_with_real_sessions, fake_llm_two_tool_calls,
    add_message_spy,
):
    """The legacy BufferingRecorder.add_message drain path inside
    _compose_loop is deleted. add_message is still called for
    non-tool surfaces (system messages, user turns) but never from
    inside _compose_loop's per-turn body."""
    composer_service_with_real_sessions._run_one_turn_for_test(
        llm=fake_llm_two_tool_calls
    )
    # add_message_spy records callers via inspect.stack(); none should
    # originate inside _compose_loop frames.
    callers = add_message_spy.caller_frames
    assert "_compose_loop" not in {frame.function for frame in callers}
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v -k "step2_dispatches or raw_content or legacy"
```

Expected: FAIL — `persist_compose_turn_async` is not yet called from `_compose_loop`; the legacy `add_message` drain path is still present.

- [ ] **Step 2: Capture `raw_content` before runtime preflight rewrites assistant content.**

Spec §5.2.1 line 1489 sets `raw_content=raw_assistant_content`. The "raw" is the pre-redaction LLM output; existing routes already pass `raw_content=result.raw_assistant_content` to `add_message` at the route layer. Inside `_compose_loop`, the equivalent value must be captured **before** any runtime preflight rewrite mutates `assistant_message.content`.

Read `web/composer/service.py` around the LLM-call site to confirm the precise location where `assistant_message.content` is first available. Bind:

```python
raw_assistant_content = assistant_message.content or ""
# (any subsequent preflight rewrites mutate assistant_message.content
# but raw_assistant_content retains the original.)
```

Hold the variable through Step 1's await boundary so it is available at Step 2 dispatch time.

- [ ] **Step 3: Replace the legacy persistence with the single dispatch.**

After Task 5's redaction step, insert:

```python
# Step 2b — single sync dispatch via the protocol async wrapper.
# The async wrapper opens an asyncio.shield-wrapped worker thread and
# runs the sync primitive under it (commit-wins cancellation contract,
# §5.2.2). Calling self.sessions_service.persist_compose_turn directly
# from this coroutine raises RuntimeError via the async-loop guard.
audit_outcome = await self.sessions_service.persist_compose_turn_async(
    session_id=session_id,
    assistant_content=assistant_message.content or "",
    raw_content=raw_assistant_content,
    redacted_assistant_tool_calls=redacted_assistant_tool_calls,
    redacted_tool_rows=redacted_tool_rows,
    parent_composition_state_id=current_state_id,
    expected_current_state_id=current_state_id,
    writer_principal="compose_loop",
    plugin_crash_pending=plugin_crash is not None,
)
```

`current_state_id` was captured before the LLM call (Task 4 Step 2); it serves as both the parent state id and the stale-state guard. The sync primitive re-reads `MAX(version)` under the session write lock and raises `StaleComposeStateError` if a concurrent writer has advanced the session between LLM call and persist.

Delete the legacy post-loop `BufferingRecorder` drain inside `_compose_loop`. Confirm by grep:

```bash
grep -n "_persist_tool_invocations\|add_message" src/elspeth/web/composer/service.py
```

The grep should show `add_message` only in non-loop sites (system message setup, user turn entry) and `_persist_tool_invocations` should be deleted from `_compose_loop`'s call path entirely. Calls from outside `_compose_loop` (route-level user/system message persistence) are unchanged.

- [ ] **Step 4: Re-run the Step 2 tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_persistence.py -v
```

Expected: PASS for every Step 1 and Step 2 case including the plugin-crash test from Task 4 (now that the audit write runs before the raise).

- [ ] **Step 5: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 2 — single sync dispatch via persist_compose_turn_async; delete legacy drain (composer-progress-persistence phase 3)"
```

---

## Task 7: `_compose_loop` Step 3 — `AuditOutcome` + `AuditIntegrityError` dispatch

The post-dispatch step:

1. If `plugin_crash is not None`, raise the captured `ComposerPluginCrashError`. The audit row has already committed (or `audit_outcome.unwind_audit_failed=True` means the audit failed under primacy and the counter has already incremented inside `persist_compose_turn`). Either way, the plugin-crash signal is the dominant outward signal.
2. If `plugin_crash is None` and `audit_outcome.assistant_id is None`, that means `unwind_audit_failed=True` arose with no in-flight plugin crash — but that combination is structurally impossible: `unwind_audit_failed` is only set when `plugin_crash_pending=True` (spec §5.2.2 line 957–972). Treat the case as a Tier-1 invariant: raise `AuditIntegrityError`.
3. `AuditIntegrityError` raised inside the sync worker (CL-PP-4b, CL-PP-10a) propagates out of `persist_compose_turn_async` unmodified. It is in `TIER_1_ERRORS`; the surrounding `_compose_loop` MUST NOT catch it.

The third historical `_AuditOutcome.tier1_violation` shape is removed (spec §5.2.1 lines 1503–1513 — the rev-4 flag-return pattern is supplanted by raise-from-worker).

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (`_compose_loop` body, after Task 6)
- Create: `tests/unit/web/composer/test_audit_failure_primacy.py`

- [ ] **Step 1: Write the failing red test.**

Create `tests/unit/web/composer/test_audit_failure_primacy.py`:

```python
"""Audit-failure primacy disposition (spec §5.2.2 / §8.1 audit-failure
primacy test surface)."""
from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.composer.protocol import ComposerPluginCrashError


def test_tool_success_audit_commit_failure_raises_AuditIntegrityError(
    composer_service_with_real_sessions, inject_commit_OperationalError,
):
    """Tool succeeds, audit COMMIT fails — AuditIntegrityError raised
    from the sync worker; composer.audit.tool_row_tier1_violation_total
    increments; _compose_loop propagates unmodified."""
    inject_commit_OperationalError()  # SQLAlchemy event hook
    with pytest.raises(AuditIntegrityError) as excinfo:
        composer_service_with_real_sessions._run_one_turn_for_test(...)
    # chained from the OperationalError:
    assert isinstance(excinfo.value.__cause__, OperationalError)
    # counter incremented:
    counter_value = composer_service_with_real_sessions._telemetry.tool_row_tier1_violation_total
    from elspeth.web.sessions.telemetry import observed_value
    assert observed_value(counter_value) == 1


def test_tool_failure_audit_commit_failure_returns_unwind_audit_failed(
    composer_service_with_real_sessions, inject_commit_OperationalError,
    fake_llm_runtime_error_on_second,
):
    """Tool fails AND audit COMMIT fails — AuditOutcome(assistant_id=None,
    unwind_audit_failed=True) returned; unwind counter increments;
    log permitted under primacy (audit-system failure);
    ComposerPluginCrashError propagates (the dominant outward signal)."""
    inject_commit_OperationalError()
    with pytest.raises(ComposerPluginCrashError):
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_runtime_error_on_second
        )
    from elspeth.web.sessions.telemetry import observed_value
    telemetry = composer_service_with_real_sessions._telemetry
    assert observed_value(telemetry.tool_row_persist_failed_during_unwind_total) == 1
    # Tier-1 counter does NOT increment in this branch (primacy disposition):
    assert observed_value(telemetry.tool_row_tier1_violation_total) == 0


def test_audit_IntegrityError_propagates_regardless_of_in_flight_tool_state(
    composer_service_with_real_sessions, inject_IntegrityError_on_chat_messages,
    fake_llm_two_tool_calls,
):
    """IntegrityError on chat_messages always propagates — no primacy
    disposition, no recovery."""
    inject_IntegrityError_on_chat_messages()
    with pytest.raises(IntegrityError):  # NOT AuditIntegrityError
        composer_service_with_real_sessions._run_one_turn_for_test(
            llm=fake_llm_two_tool_calls
        )
```

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py -v
```

Expected: FAIL — Step 3 dispatch logic not yet present.

- [ ] **Step 2: Insert Step 3 dispatch.**

After Task 6's `audit_outcome = await ...` line:

```python
# Step 3 — dispatch by audit outcome and any pending plugin crash.
#
# AuditOutcome has two valid shapes (spec §5.2.2):
#   (1) success — assistant_id set, unwind_audit_failed=False.
#   (2) tool failed AND audit unwind failed — assistant_id=None,
#       unwind_audit_failed=True. (Only reachable when
#       plugin_crash_pending=True; see spec §5.2.2 line 957.)
#
# A third combination — assistant_id=None with plugin_crash=None — is
# structurally unreachable: persist_compose_turn raises
# AuditIntegrityError on Tier-1 failure, never returns the flag-set
# shape with no plugin crash in flight.
if plugin_crash is not None:
    # The captured plugin-bug exception is the dominant outward signal.
    # Audit either committed (audit_outcome.assistant_id set) or
    # failed under primacy (audit_outcome.unwind_audit_failed=True);
    # in either case the route-helper _handle_plugin_crash runs and
    # produces the 500 response with partial_state + failed_turn.
    raise plugin_crash
# Success path: turn committed atomically. Loop continues.
```

The `assert audit_outcome.assistant_id is not None` style invariant check is intentionally absent — Python will surface the structural impossibility via `AttributeError` if a future change to `AuditOutcome` breaks it, which is the offensive-programming posture. A defensive `if audit_outcome.assistant_id is None: raise ...` here would mask a Phase 1 contract violation.

- [ ] **Step 3: Re-run the audit-failure-primacy tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_audit_failure_primacy.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit.**

```bash
git commit -am "feat(composer): _compose_loop Step 3 — AuditOutcome dispatch + plugin-crash raise-after-audit (composer-progress-persistence phase 3)"
```

---

## Task 8: `failed_turn` response field on `_handle_*` helpers

Add `failed_turn` to the response body of `_handle_convergence_error`, `_handle_plugin_crash`, and `_handle_runtime_preflight_failure` (web/sessions/routes.py:998, :1136, :1292). The field carries `assistant_message_id`, `tool_calls_attempted`, `tool_responses_persisted`, and `transcript_url`. Phase 4 consumes this to decide whether the recovery panel opens.

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py` (three handler functions)
- Create: `tests/integration/web/test_compose_loop_failed_turn_field.py`

- [ ] **Step 1: Write the failing red test.**

In `tests/integration/web/test_compose_loop_failed_turn_field.py`:

```python
"""failed_turn response shape across the three _handle_* helpers
(spec §6.1)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_handle_convergence_error_returns_failed_turn(
    composer_test_client, fake_llm_exceeds_budget,
):
    """Convergence error mid-loop → failed_turn carries
    assistant_message_id, tool_calls_attempted, tool_responses_persisted,
    transcript_url."""
    response = composer_test_client.post("/api/sessions/.../compose",
                                          json={...})
    assert response.status_code == 422
    body = response.json()
    assert "failed_turn" in body
    ft = body["failed_turn"]
    assert "assistant_message_id" in ft
    assert isinstance(ft["tool_calls_attempted"], int)
    assert isinstance(ft["tool_responses_persisted"], int)
    assert ft["tool_responses_persisted"] <= ft["tool_calls_attempted"]
    assert ft["transcript_url"].endswith("&include_tool_rows=true")


def test_handle_plugin_crash_returns_failed_turn(
    composer_test_client, fake_llm_plugin_crash,
):
    """Plugin crash mid-loop → 500 response carries failed_turn."""
    response = composer_test_client.post("/api/sessions/.../compose",
                                          json={...})
    assert response.status_code == 500
    body = response.json()
    assert "failed_turn" in body


def test_handle_runtime_preflight_failure_returns_failed_turn(
    composer_test_client, fake_llm_preflight_blocks,
):
    """Runtime preflight failure → response carries failed_turn (with
    tool_responses_persisted=0 since no tools executed)."""
    response = composer_test_client.post("/api/sessions/.../compose",
                                          json={...})
    body = response.json()
    assert body["failed_turn"]["tool_responses_persisted"] == 0
```

```bash
.venv/bin/python -m pytest tests/integration/web/test_compose_loop_failed_turn_field.py -v
```

Expected: FAIL — none of the helpers emits `failed_turn` yet.

- [ ] **Step 2: Add the field to each helper.**

In each of `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure`, after `partial_state` is computed and before `response_body` is returned:

```python
tool_responses_persisted = await sessions_service.count_tool_responses_for_assistant_async(
    assistant_message_id=assistant_message_id_or_none,  # None if no assistant row landed
)
tool_calls_attempted = len(assistant_message.tool_calls) if assistant_message else 0

response_body["failed_turn"] = {
    "assistant_message_id": assistant_message_id_or_none,
    "tool_calls_attempted": tool_calls_attempted,
    "tool_responses_persisted": tool_responses_persisted,
    "transcript_url": (
        f"/api/sessions/{session_id}/messages"
        f"?since={user_message_id}&include_tool_rows=true"
    ),
}
```

Task 9 below adds `count_tool_responses_for_assistant_async`; this task wires the call site.

- [ ] **Step 3: Re-run the test (after Task 9 lands the helper).**

Expected: PASS once Task 9 is in.

- [ ] **Step 4: Commit (combined with Task 9's helper).**

This task's commit is paired with Task 9 because the route-layer wiring is meaningless without the helper. See Task 9 Step 5.

---

## Task 9: `count_tool_responses_for_assistant` read helper on SessionsService

`failed_turn.tool_responses_persisted` is computed by a SELECT after the audit writes commit. The route helper does not run inline SQL; the SELECT lives on `SessionsService` as a typed read helper. Spec §6.1 read-consistency note: this SELECT runs after the surrounding compose-loop coroutine has fully awaited, so the writes are durable.

**Files:**
- Modify: `src/elspeth/web/sessions/service.py` (add `count_tool_responses_for_assistant` sync + `count_tool_responses_for_assistant_async` async dispatcher)
- Modify: `src/elspeth/web/sessions/protocol.py` (add the async method to `SessionServiceProtocol`)
- Create: `tests/unit/web/sessions/test_count_tool_responses_for_assistant.py`

- [ ] **Step 1: Write the failing red test.**

```python
"""count_tool_responses_for_assistant — read helper used by the route
layer to compute failed_turn.tool_responses_persisted (spec §6.1)."""
from __future__ import annotations

import pytest

from elspeth.web.sessions.engine import (
    create_session_engine, initialize_session_schema,
)
from sqlalchemy.pool import StaticPool


@pytest.fixture
def sessions_service():
    engine = create_session_engine("sqlite://", poolclass=StaticPool)
    initialize_session_schema(engine)
    return SessionServiceImpl(engine=engine, ...)


def test_count_zero_when_no_tool_rows(sessions_service, persisted_assistant_no_tools):
    count = sessions_service.count_tool_responses_for_assistant(
        assistant_message_id=persisted_assistant_no_tools.id,
    )
    assert count == 0


def test_count_matches_inserted_tool_rows(sessions_service, persisted_assistant_three_tools):
    count = sessions_service.count_tool_responses_for_assistant(
        assistant_message_id=persisted_assistant_three_tools.id,
    )
    assert count == 3


def test_count_none_assistant_id_returns_zero(sessions_service):
    """Assistant row never landed (e.g., CL-PP-4a) → count is 0."""
    count = sessions_service.count_tool_responses_for_assistant(
        assistant_message_id=None,
    )
    assert count == 0


@pytest.mark.asyncio
async def test_async_dispatcher_runs_in_worker_thread(sessions_service, persisted_assistant_three_tools):
    count = await sessions_service.count_tool_responses_for_assistant_async(
        assistant_message_id=persisted_assistant_three_tools.id,
    )
    assert count == 3
```

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_count_tool_responses_for_assistant.py -v
```

Expected: FAIL — method does not exist.

- [ ] **Step 2: Add the sync helper.**

In `src/elspeth/web/sessions/service.py`, after the existing read helpers:

```python
def count_tool_responses_for_assistant(
    self,
    *,
    assistant_message_id: str | None,
) -> int:
    """Count role='tool' rows linked to the given assistant message.

    Used by the route layer to compute
    ``failed_turn.tool_responses_persisted`` after the compose-loop
    coroutine has fully awaited (spec §6.1 read-consistency note —
    persist_compose_turn's writes are durable by the time this SELECT
    runs because the route helper awaits the full compose-loop
    coroutine before calling this).

    ``assistant_message_id=None`` (no assistant row landed, e.g.
    CL-PP-4a) returns 0. The check is at the boundary, not inside the
    SQL, to keep the predicate path simple.
    """
    if assistant_message_id is None:
        return 0
    with self._engine.connect() as conn:
        result = conn.execute(
            select(func.count()).select_from(chat_messages_table)
            .where(chat_messages_table.c.parent_assistant_id == assistant_message_id)
            .where(chat_messages_table.c.role == "tool")
        ).scalar_one()
    return int(result)
```

- [ ] **Step 3: Add the async dispatcher.**

In `src/elspeth/web/sessions/service.py`:

```python
async def count_tool_responses_for_assistant_async(
    self,
    *,
    assistant_message_id: str | None,
) -> int:
    return await self._run_sync(
        self.count_tool_responses_for_assistant,
        assistant_message_id=assistant_message_id,
    )
```

In `src/elspeth/web/sessions/protocol.py`, add the method to `SessionServiceProtocol` so the route layer types it via the protocol, not the concrete class.

- [ ] **Step 4: Re-run Task 8 + Task 9 tests.**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/sessions/test_count_tool_responses_for_assistant.py \
  tests/integration/web/test_compose_loop_failed_turn_field.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit (paired with Task 8's wiring).**

```bash
git commit -am "feat(sessions): count_tool_responses_for_assistant + failed_turn response field (composer-progress-persistence phase 3)"
```

---

## Task 10: `include_tool_rows` query parameter on `GET /api/sessions/{sid}/messages`

Extend the existing messages endpoint with the new query parameter and the new response-row fields. Default `include_tool_rows=false` preserves the live chat panel's existing behaviour; `include_tool_rows=true` returns tool rows interleaved by `sequence_no` AND triggers the audit-grade access-log emission in Task 11.

**Files:**
- Modify: `src/elspeth/web/sessions/routes.py` (the messages-list endpoint)
- Modify: `src/elspeth/web/sessions/schemas.py` (message response model — add `tool_call_id`, `parent_assistant_id`, `sequence_no`)
- Create: `tests/unit/web/sessions/test_messages_route_include_tool_rows.py`

- [ ] **Step 1: Write the failing red test.**

```python
"""GET /api/sessions/{sid}/messages — include_tool_rows query parameter
(spec §6.2)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def test_default_excludes_tool_rows(test_client, session_with_user_assistant_tool_rows):
    response = test_client.get(f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages")
    assert response.status_code == 200
    rows = response.json()["messages"]
    assert all(r["role"] in ("user", "assistant", "system") for r in rows)


def test_include_tool_rows_returns_tool_rows_interleaved_by_sequence_no(
    test_client, session_with_user_assistant_tool_rows,
):
    response = test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    assert response.status_code == 200
    rows = response.json()["messages"]
    sequence_nos = [r["sequence_no"] for r in rows]
    assert sequence_nos == sorted(sequence_nos)  # strictly monotonic
    roles = [r["role"] for r in rows]
    assert "tool" in roles


def test_response_rows_expose_new_columns(
    test_client, session_with_user_assistant_tool_rows,
):
    response = test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    rows = response.json()["messages"]
    for row in rows:
        assert "sequence_no" in row
        if row["role"] == "tool":
            assert row["tool_call_id"] is not None
            assert row["parent_assistant_id"] is not None
        else:
            assert row.get("tool_call_id") is None
            assert row.get("parent_assistant_id") is None
```

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_messages_route_include_tool_rows.py -v
```

Expected: FAIL.

- [ ] **Step 2: Extend the response schema.**

In `src/elspeth/web/sessions/schemas.py`, the existing message response model gains three optional fields:

```python
class MessageResponse(BaseModel):
    # ... existing fields ...
    tool_call_id: str | None = None
    parent_assistant_id: str | None = None
    sequence_no: int  # NOT None — every row has one
```

- [ ] **Step 3: Extend the endpoint.**

```python
@router.get("/api/sessions/{session_id}/messages")
async def list_messages(
    session_id: str,
    include_tool_rows: bool = Query(False, description=(
        "When true, role='tool' rows are interleaved by sequence_no. "
        "Triggers audit-grade access logging per spec §6.3."
    )),
    since: str | None = Query(None),
    sessions_service: SessionServiceProtocol = Depends(get_sessions_service),
    request: Request = ...,
    principal: AuthPrincipal = Depends(...),
) -> MessagesListResponse:
    # Existing ownership check is unchanged.
    await _enforce_session_ownership(...)

    # Task 11 adds the audit-grade-view access-log emission here when
    # include_tool_rows=true. Insertion point marked.

    messages = await sessions_service.list_messages_async(
        session_id=session_id,
        include_tool_rows=include_tool_rows,
        since=since,
    )
    return MessagesListResponse(messages=messages)
```

`SessionServiceImpl.list_messages` already exists for the default case — extend it to accept `include_tool_rows: bool` and conditionally widen the role filter. Default ordering is `(sequence_no ASC)` per spec §6.2.

- [ ] **Step 4: Re-run the test.**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_messages_route_include_tool_rows.py -v
```

Expected: PASS for the three default-vs-true tests. The audit-grade-view assertion belongs to Task 11.

- [ ] **Step 5: Commit.**

```bash
git commit -am "feat(sessions): include_tool_rows query parameter on messages endpoint (composer-progress-persistence phase 3)"
```

---

## Task 11: `record_audit_grade_view` + `audit_access_log` emission

When `include_tool_rows=true`, emit an `audit_access_log` row (writer_principal='audit_grade_view') before returning rows; increment `composer.audit.audit_grade_view_total`. The table schema is already in place from Phase 1 (`src/elspeth/web/sessions/models.py:478`).

**Files:**
- Modify: `src/elspeth/web/sessions/service.py` (add `record_audit_grade_view` + async dispatcher)
- Modify: `src/elspeth/web/sessions/protocol.py` (add `record_audit_grade_view_async` to the protocol)
- Modify: `src/elspeth/web/sessions/telemetry.py` (`_SessionsTelemetry` gains `audit_grade_view_total`; build_sessions_telemetry wires it)
- Modify: `src/elspeth/web/sessions/routes.py` (the messages endpoint emits the row before returning when `include_tool_rows=true`)
- Create: `tests/unit/web/sessions/test_record_audit_grade_view.py`

- [ ] **Step 1: Write the failing red test.**

```python
"""record_audit_grade_view — write-helper for the audit_access_log
table (spec §6.3)."""
from __future__ import annotations

import pytest


def test_record_audit_grade_view_writes_row(sessions_service, session_owned_by_alice):
    sessions_service.record_audit_grade_view(
        session_id=session_owned_by_alice.id,
        requesting_principal="alice",
        request_path="/api/sessions/sess-1/messages",
        query_args={"include_tool_rows": "true", "since": "msg-42"},
        ip_address="10.0.0.5",
    )
    rows = sessions_service.list_audit_access_log(session_id=session_owned_by_alice.id)
    assert len(rows) == 1
    row = rows[0]
    assert row.requesting_principal == "alice"
    assert row.writer_principal == "audit_grade_view"
    assert row.request_path == "/api/sessions/sess-1/messages"
    assert row.query_args == {"include_tool_rows": "true", "since": "msg-42"}
    assert row.ip_address == "10.0.0.5"


def test_record_audit_grade_view_increments_counter(sessions_service, session_owned_by_alice):
    from elspeth.web.sessions.telemetry import observed_value
    sessions_service.record_audit_grade_view(
        session_id=session_owned_by_alice.id,
        requesting_principal="alice",
        request_path="/api/sessions/sess-1/messages",
        query_args={},
        ip_address=None,
    )
    assert observed_value(sessions_service._telemetry.audit_grade_view_total) == 1


def test_record_audit_grade_view_writer_principal_is_pinned(sessions_service, session_owned_by_alice):
    """The helper does NOT accept a writer_principal argument — pinned
    to 'audit_grade_view'. Admin-tool writes use a separate code path."""
    import inspect
    sig = inspect.signature(sessions_service.record_audit_grade_view)
    assert "writer_principal" not in sig.parameters


def test_endpoint_emits_audit_log_when_include_tool_rows_true(
    test_client, session_with_user_assistant_tool_rows,
):
    test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
        f"?include_tool_rows=true"
    )
    sessions_service = test_client.app.state.sessions_service
    rows = sessions_service.list_audit_access_log(
        session_id=session_with_user_assistant_tool_rows.id,
    )
    assert len(rows) == 1


def test_endpoint_does_not_emit_audit_log_when_include_tool_rows_false(
    test_client, session_with_user_assistant_tool_rows,
):
    test_client.get(
        f"/api/sessions/{session_with_user_assistant_tool_rows.id}/messages"
    )
    sessions_service = test_client.app.state.sessions_service
    rows = sessions_service.list_audit_access_log(
        session_id=session_with_user_assistant_tool_rows.id,
    )
    assert rows == []
```

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_record_audit_grade_view.py -v
```

Expected: FAIL.

- [ ] **Step 2: Add the counter to `_SessionsTelemetry`.**

In `src/elspeth/web/sessions/telemetry.py`, extend the dataclass and the build function:

```python
audit_grade_view_total: _Counter
# ...
audit_grade_view_total=_FakeCounter(),
# ...
audit_grade_view_total=meter.create_counter("composer.audit.audit_grade_view_total"),
```

Update the telemetry.py:126–132 docstring to drop the "Phase 3 (compose loop + audit-grade view) adds … `audit_grade_view_total`" forward-looking line.

- [ ] **Step 3: Add `record_audit_grade_view` to SessionsService.**

```python
def record_audit_grade_view(
    self,
    *,
    session_id: str,
    requesting_principal: str,
    request_path: str,
    query_args: Mapping[str, str],
    ip_address: str | None,
) -> None:
    """Append one row to audit_access_log (spec §6.3).

    The writer_principal is pinned to 'audit_grade_view'; admin-tool
    writes go through a separate path. The audit_access_log table is
    append-only — there is no update or delete API exposed by
    SessionsService.

    Called from the messages endpoint when include_tool_rows=true.
    The Phase 1 schema (models.py:478) already defines the table
    with the writer_principal CHECK constraint pinning the value to
    one of ('audit_grade_view', 'admin_tool').
    """
    now = self._now()
    with self._engine.begin() as conn:
        conn.execute(
            audit_access_log_table.insert().values(
                id=self._new_id(),
                timestamp=now,
                session_id=session_id,
                requesting_principal=requesting_principal,
                request_path=request_path,
                query_args=dict(query_args),  # JSON-serialisable
                ip_address=ip_address,
                writer_principal="audit_grade_view",
            )
        )
    self._telemetry.audit_grade_view_total.add(1)

async def record_audit_grade_view_async(
    self,
    *,
    session_id: str,
    requesting_principal: str,
    request_path: str,
    query_args: Mapping[str, str],
    ip_address: str | None,
) -> None:
    return await self._run_sync(
        self.record_audit_grade_view,
        session_id=session_id,
        requesting_principal=requesting_principal,
        request_path=request_path,
        query_args=query_args,
        ip_address=ip_address,
    )
```

Add `record_audit_grade_view_async` to `SessionServiceProtocol` so the route layer types via the protocol.

- [ ] **Step 4: Wire the emission into the messages endpoint.**

In the `list_messages` route handler from Task 10, at the marked insertion point:

```python
if include_tool_rows:
    # Audit-grade access logging (spec §6.3). Emit BEFORE returning
    # rows so the log row is durable even if the response is dropped
    # mid-stream. Audit primacy: the log fires synchronously; if it
    # fails, the request fails — we do not return tool rows whose
    # access went unrecorded.
    await sessions_service.record_audit_grade_view_async(
        session_id=session_id,
        requesting_principal=principal.subject,
        request_path=request.url.path,
        query_args={k: v for k, v in request.query_params.items()},
        ip_address=request.client.host if request.client else None,
    )
```

- [ ] **Step 5: Re-run the tests.**

```bash
.venv/bin/python -m pytest tests/unit/web/sessions/test_record_audit_grade_view.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git commit -am "feat(sessions): record_audit_grade_view + audit-grade access logging (composer-progress-persistence phase 3)"
```

---

## Task 12: Integration test surface — CL-PP-* scenarios

This task closes the §11 done-when checklist by ensuring every CL-PP-* scenario from §8.2 passes against `test_composer_llm_eval_characterization.py` and (for CL-PP-11) the testcontainer PostgreSQL CI lane.

### Task 12a: Re-baseline CL-PP-1..8 for the new response shape

These scenarios exist in the characterization test surface (some in skeletal pre-Phase-3 form). They each need updating to assert the new `failed_turn` field on the response body and the new per-row columns on transcript responses.

- [ ] **Step 1: Audit `tests/integration/pipeline/test_composer_llm_eval_characterization.py` for existing CL-PP-1..8 cases.**

```bash
grep -n "CL-PP-1\|CL-PP-2\|CL-PP-3\|CL-PP-4a\|CL-PP-4b\|CL-PP-4c\|CL-PP-4d\|CL-PP-5\|CL-PP-6\|CL-PP-7\|CL-PP-8" tests/integration/pipeline/test_composer_llm_eval_characterization.py
```

For each existing case:

- If it asserts only the legacy response shape, extend it to also assert `failed_turn` per §6.1.
- If the case predates the `composition_states.provenance` discriminator, add the discriminator check.
- If the case is missing (CL-PP-4d, e.g.), author it new.

- [ ] **Step 2: Each case follows the same TDD cycle.**

Run the case → confirm RED on the new assertions → update the implementation only if a defect surfaces (the loop should now pass) → confirm GREEN.

- [ ] **Step 3: Commit (one commit per scenario or per coherent batch).**

```bash
git commit -am "test(integration): CL-PP-{N} extends failed_turn assertions (composer-progress-persistence phase 3)"
```

### Task 12b: Author CL-PP-9, 10a, 10b, 12, 13

- [ ] **CL-PP-9: Mixed redaction policy (§8.2 line 2583).** A tool whose argument model has both `Sensitive[T]`-annotated and non-sensitive fields. Drive a turn that exercises both. Assert sensitive fields persist as sentinel/summarizer output; non-sensitive fields are byte-identical; structural shape preserved.

- [ ] **CL-PP-10a: INSERT succeeded, COMMIT failed (no plugin crash).** Inject `OperationalError` on COMMIT via SQLAlchemy event hook. Assert `AuditIntegrityError` raised (chained from the injected error); `composer.audit.tool_row_tier1_violation_total` increments; caller propagates; no rows visible (transaction rolled back).

- [ ] **CL-PP-10b: COMMIT failed (plugin crash in flight).** Same injection plus a `RuntimeError` from the second tool. Assert `AuditOutcome(assistant_id=None, unwind_audit_failed=True)`; `tool_row_persist_failed_during_unwind_total` increments; log entry emitted; caller raises the captured `ComposerPluginCrashError`.

- [ ] **CL-PP-12: Tool-call cap exceeded.** LLM emits 17 tool calls; loop raises `ComposerConvergenceError(reason="tool_call_cap_exceeded")` BEFORE any tool execution; no DB writes; `composer.tool_call_cap_exceeded_total` increments.

- [ ] **CL-PP-13: Unknown response key fail-closed (§8.2 line 2627).** A declarative-manifest-entry tool returns a response containing a key not in `known_response_keys`. Assert the value is replaced with the fixed sentinel `<redacted-unknown-response-key>` (rev-5 form; no length disclosure); `composer.redaction.unknown_response_key_total` increments. The tool call completes successfully.

Each case follows: RED test → run → GREEN. Commit per case.

### Task 12c: Property test extension + schema-level backward-direction test

- [ ] **Step 1: Author the strategy contracts at `tests/property/web/composer/strategies.py`.**

Strategies per spec §8.3.1: `st_tool_call`, `st_argument_dict`, `st_redaction_policy`, `st_failure_injection_point` (with `audit_raises_OperationalError_on_commit`, `advisory_lock_unavailable`, `tool_call_cap_exceeded`, `unknown_response_key` arms), `st_cancellation_arrival_time` (with the rev-4-expanded enum covering `during_run_sync_between_insert_and_commit`, `during_advisory_lock_acquisition`, `after_commit_before_response_yielded`), `st_session_state`. Use Hypothesis `@example(...)` decorators on the failure-injection enum to guarantee every branch is reached (closes spec QA F-6). The strategies file's docstring contains a mapping table from §5.5 row numbers to strategy values so future drift is detectable.

- [ ] **Step 2: Author the stateful machine at `tests/property/web/composer/test_compose_loop_invariants.py`.**

Uses Hypothesis's `RuleBasedStateMachine`. After each trace, the machine asserts the §8.3.2 post-conditions: forward-direction, backward-direction (via the schema-level SQL predicate), ordering & uniqueness, redaction, cancellation-specific, audit-failure primacy, OTel counter post-conditions.

- [ ] **Step 3: Author the schema-level integration test at `tests/integration/web/test_inv_audit_ahead_backward.py`.**

```python
"""INV-AUDIT-AHEAD bidirectional schema-level test (spec §4.1.2 /
§5.3 / §8.3.2 closes QA F-1)."""
from __future__ import annotations

from sqlalchemy import text


def test_no_state_row_without_tool_row(populated_audit_db):
    """The post-condition is now a pure SQL predicate."""
    with populated_audit_db.connect() as conn:
        result = conn.execute(text(
            "SELECT cs.id FROM composition_states cs "
            "LEFT JOIN chat_messages cm "
            "  ON cm.composition_state_id = cs.id AND cm.role = 'tool' "
            "WHERE cs.provenance = 'tool_call' AND cs.version > 0 "
            "  AND cm.id IS NULL"
        ))
        orphans = result.fetchall()
    assert orphans == []
```

The fixture `populated_audit_db` exercises the compose loop end-to-end with mixed tool successes/failures/cancellations so the predicate has meaningful surface to evaluate.

- [ ] **Step 4: Run the full property + integration surface.**

```bash
.venv/bin/python -m pytest tests/property/web/composer/ tests/integration/web/test_inv_audit_ahead_backward.py -v
```

Expected: PASS. The property test runs the §8.3.2 OTel-counter post-conditions across the campaign.

- [ ] **Step 5: Commit.**

```bash
git commit -am "test(integration): CL-PP-9/10/12/13 + property + schema-level backward-direction (composer-progress-persistence phase 3)"
```

---

## Task 13: OTel alert routes, dashboard, runbook for Tier-1 audit counters

Overview done-when item 5: "before Phase 3 ships the Tier-1 audit counters have an alert route, dashboard visibility, and runbook entry."

`composer.audit.tool_row_tier1_violation_total` is the critical counter — non-zero on any production trace indicates a Tier-1 audit-write failure. `composer.audit.tool_row_integrity_violation_total` is the second-critical counter — non-zero indicates a schema-constraint breach (writer_principal misuse, sequence-no collision, etc.). The three remaining counters (`audit_grade_view_total`, `tool_row_persist_failed_during_unwind_total`, `tool_call_cap_exceeded_total`) are operational signals, not Tier-1 invariants.

**Files:**
- Inspect: `config/` and `infra/` directories for existing alert/dashboard plumbing
- Modify or create: alert config, dashboard config, runbook entries — exact files depend on existing infra

- [ ] **Step 1: Survey existing alert/dashboard infra.**

```bash
find config infra -type f \( -name '*.yml' -o -name '*.yaml' -o -name '*.json' \) 2>/dev/null \
  | xargs grep -l "composer\.\|sessions\.\|alertmanager\|grafana" 2>/dev/null
ls docs/runbooks/ 2>/dev/null
```

If existing infra is found, append the new alert routes + dashboard panel + runbook entry to the existing files. If no existing infra is found, this task **scopes down**:

  - Wire the counters in `_SessionsTelemetry` (Tasks 2 + 11 already do this — confirm).
  - File three follow-up Filigree tickets for the alert route, dashboard, and runbook deliverables; capture the ticket IDs.
  - Cite the three ticket IDs in the Phase 3 PR description with the heading "Pre-merge prerequisites for production deploy" — the operator confirms the tickets are tracked before merge but they do not block the PR review.

The overview done-when requires that the artifacts exist before Phase 3 **ships** (production deploy), not before the PR merges. The split lets Phase 3 land in main while the operational plumbing follows; the merge to a release branch (RC5 or successor) is the gate.

- [ ] **Step 2: Commit (or comment in PR if scoped to tickets).**

```bash
git commit -am "ops(audit): alert routes / dashboard / runbook entries for Tier-1 audit counters (composer-progress-persistence phase 3)"
```

Or, if scoped to follow-up tickets, no commit — just the PR-description block citing the ticket IDs.

---

## Task 14: Overview update — rev-5 / manifest-keyed framing

The overview at `docs/superpowers/plans/2026-04-30-composer-progress-persistence-overview.md` still references rev-4 / `Sensitive[T]` in five places: lines 5, 9, 20, 27, 29. Phase 3 ships the file the overview indexed; update the overview to keep the doc set internally consistent.

**Files:**
- Modify: `docs/superpowers/plans/2026-04-30-composer-progress-persistence-overview.md`

- [ ] **Step 1: Update the spec-revision pointer.**

Line 5: "revision 4, 2481 lines" → "revision 5, 3608 lines (manifest-keyed redaction)".

- [ ] **Step 2: Update the architecture summary.**

Line 9: "Phase 2 adds the type-driven redaction primitive (`Sensitive[T]`) plus the legacy declarative escape valve" → "Phase 2 adds the manifest-keyed redaction primitive (`ToolRedaction` dataclass + module-level `MANIFEST`), promotes ~6–8 sensitive-touching tools to type-driven `argument_model` entries via `Sensitive[T]` annotations, and retains the declarative `ToolRedactionPolicy` shape for the remaining ~29–31 tools.".

- [ ] **Step 3: Update the Phase 2 row in the phase plans table.**

Line 20: rewrite to "`ToolRedaction` manifest dataclass, module-level `MANIFEST` keyed by tool name, `Sensitive[T]` promotion wave for ~6–8 tools, declarative `ToolRedactionPolicy` + `HandlesNoSensitiveDataReason` for the remaining ~29–31 tools, shared traversal iterator, `RedactionTelemetry` Protocol, four-assertion adequacy guard, broadened policy-hash snapshot, label-gate CI step.".

- [ ] **Step 4: Update the Phase 3 row.**

Line 21: keep largely as-is; the existing description is rev-5-correct (compose-loop integration, tool-call cap, failed_turn, include_tool_rows). No edit required — confirm and move on.

- [ ] **Step 5: Update the cross-phase dependencies.**

Line 27: "Phase 3 depends on Phase 2's redaction primitives (the compose loop uses `redact_tool_call` and `lookup_tool_class`)." → "Phase 3 depends on Phase 2's redaction primitives (the compose loop uses `redact_tool_call_arguments`, `redact_tool_call_response`, and the module-level `MANIFEST`; the rev-4 `lookup_tool_class` helper is removed per rev-5 §5.7.5).".

Line 29: drop the rev-4 supersession-list bullets that referred to fictional rev-4 symbols (`_StatePayload.version`, etc.); they're already handled in the spec body and the Phase 1A/1B/1C plans.

- [ ] **Step 6: Commit.**

```bash
git commit -am "docs(plan): overview reflects rev-5 manifest-keyed framing + Phase 3 plan landing (composer-progress-persistence phase 3)"
```

---

## Task 15: OQ-3 Filigree ticket for `chat_messages` integrity-hash chain

Spec §10 OQ-3 and §11 cross-phase considerations: file a Filigree ticket for the integrity-hash chain (mechanism sketched in spec §10), cite the ID in the Phase 3 PR description. The integrity-hash chain is out of scope for Phase 3 itself — only the ticket-filing is in scope.

- [ ] **Step 1: File the ticket.**

```bash
filigree create "chat_messages integrity-hash chain — composer-progress-persistence OQ-3" \
  --type=task --priority=2 \
  --label-prefix=composer-progress-persistence
```

The ticket body should reference spec §10 OQ-3 and §11; include a short summary of the mechanism (per-row hash chain seeded by the previous row's hash, anchored at the session row).

- [ ] **Step 2: Capture the ticket ID for the PR description.**

```bash
filigree show <new-ticket-id>
```

Cite the ticket ID in the Phase 3 PR description under the "Filed follow-ups" section.

- [ ] **Step 3: No commit (Filigree state is outside the repo).**

---

## Task 16: Final Phase 3 CI run + PR

- [ ] **Step 1: Run the full Phase 3 test surface.**

```bash
.venv/bin/python -m pytest \
  tests/unit/web/composer/test_compose_loop_tool_call_cap.py \
  tests/unit/web/composer/test_compose_loop_persistence.py \
  tests/unit/web/composer/test_audit_failure_primacy.py \
  tests/unit/web/sessions/test_count_tool_responses_for_assistant.py \
  tests/unit/web/sessions/test_record_audit_grade_view.py \
  tests/unit/web/sessions/test_messages_route_include_tool_rows.py \
  tests/integration/web/test_compose_loop_failed_turn_field.py \
  tests/integration/web/test_inv_audit_ahead_backward.py \
  tests/integration/pipeline/test_composer_llm_eval_characterization.py \
  tests/property/web/composer/ \
  -v
```

Expected: PASS for all. CL-PP-11 (commit `eca88974`) runs only on the Docker-enabled CI lane via the `testcontainer` marker.

- [ ] **Step 2: Run static-analysis gates.**

```bash
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/ tests/
.venv/bin/python -m scripts.check_contracts
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
.venv/bin/python scripts/cicd/enforce_freeze_guards.py check
```

Expected: all green.

- [ ] **Step 3: Verify counter post-conditions match the §1.4 SLO claims.**

```bash
.venv/bin/python -m pytest tests/property/web/composer/ -v -k "otel_counter_postconditions"
```

Expected: PASS. `composer.audit.tool_row_tier1_violation_total == 0` across the property-test campaign (the counter only increments when the test explicitly injects a Tier-1 fault and asserts the increment).

- [ ] **Step 4: Surface to operator for PR-open decision. Do NOT run `gh pr create`.**

Per Phase 2 rev-5 BLOCKER B4 closure pattern (per `docs/superpowers/plans/2026-04-30-composer-progress-persistence-phase-2-redaction.md` "Phase 2 done-when removes PR-open from scope; plan rewrite ends at 'gate green; await operator PR-open instruction'"), and per project memory `feedback_default_to_worktree.md` (worktree-default policy revision 2026-05-11) and `project_phase2_plan_review_verdict.md`: Phase 3 implementation ends at "gate green; await operator PR-open instruction." The implementer captures the readiness state in the conversation and stops; the operator decides when (and whether) to open the PR. (Rev-1 of this plan ran `gh pr create` unconditionally here; that was a re-introduction of the Phase 2 rev-1 BLOCKER B4 pattern. Rev-2 removes it.)

The implementer surfaces the following to the operator after the gate runs are green:

> Phase 3 implementation complete on `feat/composer-progress-persistence-1a` at HEAD `<commit-sha>`. Gate state:
>
> - Unit: `<N>k` pass, `<m>` skipped, `<x>` xfailed. Integration: `<M>` pass (Docker-disabled lane). Property: `<P>` pass.
> - mypy / ruff / contracts / tier-model / freeze-guards: all clean.
> - CL-PP-11 testcontainer scenario: gates on Docker-enabled CI lane only; surfaced via `pytest -m testcontainer` for the operator to spot-check locally.
>
> Phase 3 ships:
>
> - Compose-loop rewrite to the §5.2.1 single-sync-block shape: gather tool outcomes async, redact via the Phase 2 manifest walker (`MANIFEST` / `redact_tool_call_arguments` / `redact_tool_call_response` / `RedactionTelemetry`), dispatch ONE `persist_compose_turn_async` per turn; raise `ComposerPluginCrashError` after audit commit; let `AuditIntegrityError` propagate from the sync worker.
> - Per-turn tool-call cap (default 16, env-tunable via `MAX_TOOL_CALLS_PER_TURN`); new `tool_call_cap_exceeded` reason code on `ComposerConvergenceError`; new `composer.tool_call_cap_exceeded_total` counter.
> - `failed_turn` field on 422/500 response bodies emitted by `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure`; four-key shape Phase 4 consumes (`assistant_message_id`, `tool_calls_attempted`, `tool_responses_persisted`, `transcript_url`).
> - `include_tool_rows` query parameter on `GET /api/sessions/{sid}/messages`; new response-row fields `tool_call_id`, `parent_assistant_id`, `sequence_no`.
> - Audit-grade access logging via new `record_audit_grade_view` helper writing to the Phase-1-provisioned `audit_access_log` table; new `composer.audit.audit_grade_view_total` counter.
> - `count_tool_responses_for_assistant` read helper on SessionsService for the route-layer SELECT (spec §6.1 read-consistency note).
> - Legacy `BufferingRecorder.add_message` drain inside `_compose_loop` deleted (no-legacy-code policy).
> - Integration tests CL-PP-1..10b, 12, 13 (extending the existing characterization surface; CL-PP-11 already shipped in commit `eca88974`).
> - Property test (§8.3) with full strategy contracts (§8.3.1), §8.3.2 post-conditions, and the schema-level backward-direction predicate.
> - Overview updated to rev-5 / manifest-keyed framing.
>
> Open Phase 3 follow-ups filed during implementation:
>
> - OQ-3 `chat_messages` integrity-hash chain — Filigree ticket: `<id>`.
> - Pre-merge-to-release prerequisites (per overview done-when item 5): alert route, dashboard, runbook entries — Filigree tickets `<id-1>`, `<id-2>`, `<id-3>`.
>
> Ready for operator PR-open decision. The PR description body (below) is pre-drafted from this plan's Summary section; the operator runs `gh pr create` with the desired base and review pool when ready.

**Pre-drafted PR body (for operator use; do NOT invoke `gh pr create` from this step):**

```text
## Summary

Phase 3 of composer-progress-persistence (spec §11):

- Compose-loop rewrite to the §5.2.1 single-sync-block shape: gather tool outcomes async, redact via the Phase 2 manifest walker, dispatch ONE `persist_compose_turn_async` per turn; raise `ComposerPluginCrashError` after audit commit; let `AuditIntegrityError` propagate from the sync worker.
- Per-turn tool-call cap (default 16, env-tunable via `MAX_TOOL_CALLS_PER_TURN`); new `tool_call_cap_exceeded` reason code on `ComposerConvergenceError`; new `composer.tool_call_cap_exceeded_total` counter.
- `failed_turn` field on 422/500 response bodies emitted by `_handle_convergence_error`, `_handle_plugin_crash`, `_handle_runtime_preflight_failure`; carries `assistant_message_id`, `tool_calls_attempted`, `tool_responses_persisted`, `transcript_url`.
- `include_tool_rows` query parameter on `GET /api/sessions/{sid}/messages`; new response-row fields `tool_call_id`, `parent_assistant_id`, `sequence_no`.
- Audit-grade access logging via new `record_audit_grade_view` helper writing to the Phase-1-provisioned `audit_access_log` table; new `composer.audit.audit_grade_view_total` counter.
- `count_tool_responses_for_assistant` read helper on SessionsService for the route-layer SELECT (spec §6.1 read-consistency note).
- Legacy `BufferingRecorder.add_message` drain inside `_compose_loop` deleted (no-legacy-code policy).
- Integration tests CL-PP-1..10b, 12, 13 (extending the existing characterization surface; CL-PP-11 already shipped in commit `eca88974`).
- Property test (§8.3) with full strategy contracts (§8.3.1), §8.3.2 post-conditions, and the schema-level backward-direction predicate.
- Overview updated to rev-5 / manifest-keyed framing.

## Spec

`docs/superpowers/specs/2026-04-30-composer-progress-persistence-design.md` revision 5. Reviewer-finding traceability table at §12.2.

## Depends on

- Phase 1 PRs (1A, 1B, 1C) — merged on this branch.
- Phase 2 PR (manifest dispatch + `Sensitive[T]` promotion wave + adequacy guard + F1–F6 follow-ups) — landed at HEAD `f54ee7e8` on this branch.

## Out of scope (later phases)

- Frontend recovery panel (Phase 4).
- Integrity-hash chain (OQ-3 follow-up — Filigree ticket cited below).

## Filed follow-ups

- OQ-3 — `chat_messages` integrity-hash chain — Filigree ticket: [cite ID]
- Pre-merge-to-release prerequisites (per overview done-when item 5):
  - Alert route for `composer.audit.tool_row_tier1_violation_total` — Filigree ticket: [cite ID]
  - Grafana dashboard panel for Tier-1 audit counters — Filigree ticket: [cite ID]
  - Runbook entry for `tool_row_tier1_violation_total` and `tool_row_persist_failed_during_unwind_total` — Filigree ticket: [cite ID]

## Test plan

- [x] `test_compose_loop_tool_call_cap.py` — cap enforcement + counter
- [x] `test_compose_loop_persistence.py` — Step 1/2/3 unit surfaces
- [x] `test_audit_failure_primacy.py` — `AuditIntegrityError` vs `ComposerPluginCrashError` disposition
- [x] `test_count_tool_responses_for_assistant.py` — read helper + async dispatcher
- [x] `test_record_audit_grade_view.py` — access-log write + counter + pinned writer_principal
- [x] `test_messages_route_include_tool_rows.py` — endpoint behaviour + new response columns
- [x] `test_compose_loop_failed_turn_field.py` — `failed_turn` shape across all three `_handle_*` helpers
- [x] `test_inv_audit_ahead_backward.py` — schema-level backward-direction post-condition
- [x] `test_composer_llm_eval_characterization.py` — CL-PP-1..10b, 12, 13
- [x] `test_compose_loop_invariants.py` — property test, §8.3.2 OTel counter post-conditions
- [x] mypy / ruff / tier-model / freeze-guard / contracts CI green
- [x] Docker-enabled CI lane runs CL-PP-11; non-Docker lanes deselect via `-m "not testcontainer"`; `ci-success` aggregation job passes
```

---

## Phase 3 Done When

All 16 tasks above are complete. Specifically (closing the spec §11 done-when):

1. ✅ All §8.2 CL-PP-* scenarios pass — CL-PP-1, 2, 3, 4a, 4b, 4c, 4d, 5, 6, 7, 8, 9, 10a, 10b, 12, 13 in `test_composer_llm_eval_characterization.py`; CL-PP-11 in `test_compose_loop_concurrent_sessions.py` (Phase 1, commit `eca88974`).
2. ✅ §8.3 property test passes, including the schema-level backward-direction post-condition and the §8.3.2 OTel counter post-conditions.
3. ✅ `tests/integration/web/test_inv_audit_ahead_backward.py` schema-level backward-direction test passes.
4. ✅ `composer.audit.tool_row_tier1_violation_total == 0` across the property-test campaign (only increments when the test explicitly injects a Tier-1 fault).
5. ✅ Per-turn tool-call cap enforced before any tool execution; `composer.tool_call_cap_exceeded_total` matches injected counterexamples.
6. ✅ `failed_turn` field present on every 422/500 response from the three `_handle_*` helpers; `tool_responses_persisted` matches the actual DB state computed via `count_tool_responses_for_assistant`.
7. ✅ `include_tool_rows=true` triggers `audit_access_log` emission with `writer_principal='audit_grade_view'`; `composer.audit.audit_grade_view_total` counter matches request count.
8. ✅ Legacy `BufferingRecorder.add_message` drain inside `_compose_loop` deleted.
9. ✅ `enforce_tier_model.py`, `enforce_freeze_guards.py`, `check_contracts`, mypy, ruff CI green.
10. ✅ Overview reflects rev-5 / manifest-keyed framing.
11. ✅ OQ-3 Filigree ticket filed and cited in the PR description.
12. ✅ Pre-merge-to-release operational prerequisites filed as Filigree tickets and cited in the PR description (or merged in this PR if existing alert/dashboard infra was present).

Phase 4 begins after this PR merges. Phase 4 reads the `failed_turn` field this phase delivered and the `include_tool_rows=true` transcript endpoint this phase opened.

---

## Appendix A — Phase-2-as-shipped traceability (added rev-2)

Phase 2 shipped at HEAD `f54ee7e8` with 38 manifest entries (10 type-driven + 28 declarative) plus the F1–F6 follow-up sweep. Three deliverables from that sweep are load-bearing for Phase 3 and not obvious from the spec body — this appendix surfaces them so an implementer at task-execution time doesn't have to reverse-engineer the precedents.

### A.1 F2 — ARG_ERROR `validation_errors` field (commit `70424cc1`)

F2 added two helpers that Phase 3 inherits:

- `canonicalize_pydantic_cause(exc: BaseException | None) -> list[dict[str, Any]] | None` at `src/elspeth/web/composer/audit.py:826`. Leak-safe Pydantic-`ValidationError` canonicalisation: `loc` / `msg` / `type` only; `input` / `url` / `ctx` stripped (Tier-3 sensitive material). Returns `None` when the cause is not a `ValidationError` (recording an empty list has no audit value).
- `_arg_error_payload(exc: ToolArgumentError, tool_name: str) -> Mapping[str, Any]` at `src/elspeth/web/composer/service.py:3705`. Module-level (not loop-local) so it is testable in isolation. Builds the structured ARG_ERROR payload with two fields: `error` (operator-safe, LLM-facing) plus `validation_errors` (present iff `exc.__cause__` is a `ValidationError`). Populates `result_canonical` for the audit row AND serves as the LLM-facing `role=tool` message content.

**Implications for Phase 3:**

- Task 4's walker call on the ARG_ERROR `_ToolOutcome` payload sees a dict that may include `validation_errors`. The Phase 4 recovery panel will display these field-name details to help operators recover from Pydantic-rejected tool arguments; Phase 3's persistence must preserve the field through the redaction walker.
- The existing `except ToolArgumentError as exc:` arm in `_compose_loop` already calls `_arg_error_payload(exc, tool_name)` (current line ~L2549); Task 4's accumulator must produce the same payload shape for the `error_class="ToolArgumentError"` `_ToolOutcome` branch, otherwise the persisted record diverges from the LLM-visible one.
- §12.2 BLOCKER_A traceability — the cross-phase contract integrity guarantee at the persistence boundary that F2 closes for ARG_ERROR specifically; Phase 3's walker call is the downstream consumer of F2.

### A.2 F3 — closed-list Sensitive markers (commit `f54ee7e8`)

F3 replaced overbroad `Sensitive` markers on `routes` and `trigger` fields with closed-list types. This is a Phase 2 hardening; Phase 3 does not modify or extend it. The implication for Phase 3 is purely citation: when reviewers reference the `Sensitive[T]` annotation surface, the closed-list types are the canonical shape; the older "overbroad" form is no longer in use.

### A.3 AST-fingerprint observation `elspeth-obs-02a0002fae`

`scripts/cicd/enforce_tier_model.py` fingerprints findings by AST `body[N]` index. Any line-shifting edit to `src/elspeth/web/composer/service.py` rotates downstream fingerprints in `config/cicd/enforce_tier_model/`. F2 (commit `70424cc1`) established the mitigation pattern: append new module-level defs at file tail with a "do-not-move" header comment so the AST indices of existing definitions remain stable.

**Implications for Phase 3:**

- Task 3's `_PerTurnPersistContext` dataclass (if introduced) lands at `service.py` tail with the same "do-not-move" header. Zero allowlist refresh.
- Task 10's `record_audit_grade_view` writer lands at `web/sessions/service.py` tail with the same header. Zero refresh.
- Task 11's protocol extension appends at protocol-class tail. Minimal refresh.
- **Tasks 3/4/5/6/7 modify the `_compose_loop` body at L1668–~L2700 (current line numbers) — these edits CANNOT be tail-appended; they must land where the loop body sits.** Expected allowlist refresh: ~40–70 entries across the loop-body tasks. Refresh is mechanical (regenerate from current findings; same-commit), not investigation; the implementer must distinguish "same finding at a new fingerprint" (refresh) from "new finding" (fix the code).
- Memory: `project_tier_model_python_version` — the worktree venv MUST be Python 3.13 to avoid the ~300 spurious tier-model FPs the version-skew issue triggers.

### A.4 Current `_compose_loop` line numbers (rev-2 update)

The rev-1 plan body cited `service.py:867`, `:907`, `:942` in code-comment references inherited from the spec text. Those numbers were stale even at rev-1 authorship time; rev-2 captures the actual current locations at HEAD `f54ee7e8`:

| Surface | Rev-1 cited | Current (HEAD `f54ee7e8`) |
|---|---|---|
| `async def _compose_loop(...)` definition | `:1724`–`:1900` (approximate range) | `:1668`–~`:2700` |
| `except ToolArgumentError` (Tier-3 boundary, loop continues) | `:867` | `:2518` |
| `except (AssertionError, MemoryError, RecursionError, SystemError)` (fail-fast) | `:907` | `:2559` |
| `except Exception as tool_exc` (plugin-bug surface; `ComposerPluginCrashError.capture`) | `:942` | `:2602` |
| `_arg_error_payload(exc, tool_name)` call site inside `except ToolArgumentError` arm | not present at rev-1 | `:2549` (added by F2 `70424cc1`) |
| `_arg_error_payload` module-level helper definition | not present at rev-1 | `:3705` (added by F2; file-tail per AST-fingerprint pattern) |
| `_PROMOTED_TOOL_NAMES` frozenset (9 promoted tools excluded from `_TOOL_REQUIRED_PATHS`) | not present at rev-1 | introduced via Phase 2 promotion-wave commits |

`web/sessions/service.py` references:

| Surface | Current (HEAD `f54ee7e8`) |
|---|---|
| `def persist_compose_turn(...)` sync primitive | `:723` |
| `async def persist_compose_turn_async(...)` impl | `:980` |

`web/sessions/_persist_payload.py` (Phase 1B shipped):

| Surface | Current |
|---|---|
| `class StatePayload` | `:19` |
| `class _ToolOutcome` | `:89` |
| `class RedactedToolRow` | `:123` |
| `class AuditOutcome` | `:132` |

`web/sessions/models.py` (Phase 1A shipped):

| Surface | Current |
|---|---|
| `audit_access_log_table` (INERT; Task 11 wires writer) | `:485` |
| `writer_principal` CHECK constraint | `:502` |

`web/sessions/routes.py` (route helpers):

| Surface | Current |
|---|---|
| `_handle_convergence_error` | `:1139` |
| `_handle_plugin_crash` | `:1283` |
| `_handle_runtime_preflight_failure` | `:1433` |
| `get_messages` (the endpoint Task 10 extends) | `:2821` |

Implementers should use these current locations rather than the spec-text-inherited line numbers in the rev-1 plan body. The plan body's comments inside the inserted compose-loop code (`# service.py:867 — catches and continues...`) are kept verbatim for spec-cross-reference convenience; they reflect the spec author's intent and are not load-bearing line-precise references.

### A.5 Anticipated open questions (rev-1 inheritance — all resolved by spec rev-5)

The on-disk Phase 3 plan-authoring prompt at `notes/composer-phase-3-plan-prompt-2026-05-10.md` anticipated four "decisions to surface" before drafting. Rev-2 confirms all four are resolved by spec rev-5 and the current code; the implementer does NOT need to bring them back to the operator unless a contradiction surfaces during execution:

1. **`failed_turn.transcript_url` cursor shape.** Resolved: user-message ID. Reference: Phase 4 plan line ~140 (`?since=u_1&include_tool_rows=true`) — Phase 4's tests pin the contract.
2. **`provenance='tool_call'` write timing within the sync block.** Resolved: spec §5.2.2 step list — assistant row first, then per-tool (optional state row first, then tool row). All in one `engine.begin()` context. Implementation already at `web/sessions/service.py:723`.
3. **Latency NFR vs CI infra.** Resolved: spec §1.4 splits sanity bound (CI, p95 ≤ 250 ms) from tight target (nightly bench, p95 ≤ 25 ms). Phase 3 done-when references the sanity bound only; the tight target is filed as tracking observation per §1.4.
4. **Cancellation arrival-time enum coverage.** Resolved: spec §5.2.4 cancellation table + §8.3.1 `st_cancellation_arrival_time` nine-value enum — exhaustive coverage of the §5.5 failure-mode rows 5–11. Task 12c's strategy contract consumes this directly.

### A.6 Phase 1B `_ToolOutcome` field check (Task 1 disposition)

Task 1 of this plan is "verify `_ToolOutcome` matches spec §5.2.1." Rev-2 confirms via direct inspection at HEAD `f54ee7e8`:

- `web/sessions/_persist_payload.py:89` defines `_ToolOutcome(call: Any, response: Any, error_class: str | None, error_message: str | None, pre_version: int, post_version: int)`.
- Frozen, slots, `freeze_fields(self, "call", "response")` in `__post_init__`.
- Matches spec §5.2.1 verbatim. Task 1 is a verify-only no-op at execution time; no commit, no follow-up ticket.

If a future Phase 1 hygiene change adds or removes a field, Task 1's "Step 1: Confirm the field list" branch is the surface to catch it.

---

End of rev-2 plan, composer-progress-persistence Phase 3.
