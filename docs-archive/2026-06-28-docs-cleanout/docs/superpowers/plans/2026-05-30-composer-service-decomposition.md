# Composer `service.py` God-Class Decomposition — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose `ComposerServiceImpl` (5,313 LOC) — starting with the 1,298-line `_dispatch_tool_batch` — into focused sibling modules under `web/composer/`, behaviour-preserving, with the audit trail provably unchanged.

**Architecture:** Mirror the merged `engine/orchestrator/` pattern: `service.py` keeps the class and the thin compose driver; cohesive method-clusters become plain function-modules it invokes. The per-tool-call pipeline moves to `web/composer/tool_batch.py`, driven by an explicit frozen `ToolBatchContext` (loop-invariant inputs) + mutable `BatchAccumulator` (loop-carried state), replacing the nested closures. Sequence is strict: characterization tests pin all ~13 terminal arms FIRST, then verbatim move, then idiom collapse, then sync ToolDeclaration convergence, then the remaining oversized methods.

**Tech Stack:** Python 3.13, pytest, mypy, ruff, `elspeth-lints` tier_model; composer audit via `elspeth.contracts.composer_audit.ComposerToolStatus`; test harness in `tests/unit/web/composer/conftest.py` (`_FakeComposeLLM`, `_run_one_turn_for_test`).

**Spec:** `docs/superpowers/specs/2026-05-29-composer-service-decomposition-design.md`

**Branch:** `composer-service-decomp` (worktree off RC5.2). `git commit --no-verify` is permitted **on this branch only** (operator-granted 2026-05-29) conditioned on Tier-1 audit discipline; tier-model fingerprints are reconciled once at the end (Phase 6), not per-commit.

---

## File Structure

| File | Responsibility | Phase |
|------|----------------|-------|
| `src/elspeth/web/composer/service.py` | MODIFY — thin core; `_dispatch_tool_batch` body relocated, call site delegates | 2,5 |
| `src/elspeth/web/composer/tool_batch.py` | CREATE — per-call pipeline (`run_tool_batch`), `ToolBatchContext`, `BatchAccumulator` | 2 |
| `src/elspeth/web/composer/tool_batch_arms.py` | CREATE — uniform terminal-arm emit helpers | 3 |
| `src/elspeth/web/composer/turn_audit.py` | CREATE — `_persist_turn_audit` cluster | 5 |
| `src/elspeth/web/composer/availability.py` | CREATE — `_compute_availability` + `ComposerAvailability` | 5 |
| `src/elspeth/web/composer/no_tool_finalize.py` | CREATE — `_finalize_no_tool_response` | 5 |
| `tests/unit/web/composer/test_dispatch_arms_characterization.py` | CREATE — pins every terminal arm's audit/anti-anchor/budget side-effects | 1 |

`tool_batch.py` sits beside `service.py` (both L3 application layer); it imports from `web/composer/tools/` (also L3) — intra-layer, no upward edge. Confirm with the import-graph command in Phase 0.

---

## Phase 0 — Workspace bring-up

### Task 0: Isolated venv + green baseline

**Files:** none (environment only)

- [ ] **Step 1: Create a Python 3.13 venv in the worktree**

Run:
```bash
cd /home/john/elspeth/.worktrees/composer-service-decomp
python3.13 --version   # confirm 3.13.x is available
uv venv --python 3.13 .venv
```
Expected: `.venv` created reporting CPython 3.13. (3.13 is mandatory — `enforce_tier_model` reports ~300 spurious violations on a mismatched interpreter.)

- [ ] **Step 2: Editable install bound to THIS worktree's src**

Run:
```bash
.venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
uv pip install --python .venv/bin/python -e ".[all,dev]"
```
Expected: install completes; `elspeth` resolves to `.worktrees/composer-service-decomp/src/elspeth`. (Passing `--python` prevents the documented venv-leak that clobbers main's `.venv`.)

- [ ] **Step 3: Capture the composer-suite baseline (must be green before any change)**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/ tests/property/web/composer/ -q
```
Expected: all pass (note the count, e.g. `N passed`). If anything is red here, STOP — it is a pre-existing failure to triage before refactoring, not something this work introduced.

- [ ] **Step 4: Confirm tool_batch.py's planned home introduces no upward import edge**

Run:
```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  check --rules trust_tier.tier_model --root src/elspeth | tail -20
```
Expected: baseline finding count recorded (there may be existing allowlisted findings; note the number so Phase 6 can diff against it).

- [ ] **Step 5: No commit** (environment only).

---

## Phase 1 — Characterization tests (GATES EVERYTHING)

This phase pins the audit/anti-anchor/budget side-effects of **every** terminal arm of `_dispatch_tool_batch` BEFORE any code moves. With per-commit mechanical hooks skipped on this branch, these tests are the entire safety net (advisor: "tests-green is necessary but not sufficient" — an untested arm can lose a `recorder.record(finish_*)` and stay green).

### The terminal-arm inventory

Source: `service.py:_dispatch_tool_batch` (lines ~2374–3533). Each arm and its existing coverage:

| # | Arm | Audit status | anti_anchor | Existing fixture / test |
|---|-----|--------------|-------------|--------------------------|
| 1 | JSON-decode failure | ARG_ERROR | record_failure | **GAP** |
| 2 | non-dict arguments | ARG_ERROR | record_failure | **GAP** |
| 3 | canonicalization failure (post-decode) | ARG_ERROR | record_failure | **GAP** |
| 4 | discovery cache hit | SUCCESS (cache_hit) | (none — observation) | **GAP** |
| 5 | required-paths missing | ARG_ERROR | record_failure | partial (`misplaced_schema` is a different path) — **GAP for the `_TOOL_REQUIRED_PATHS` path** |
| 6 | approval-intercept (proposal) | SUCCESS (APPROVAL_REQUIRED) | record_success | `fake_llm_create_blob_then_set_pipeline` (covers blob-store-only exclusion + intercept) |
| 7 | advisor disabled | SUCCESS (error payload) | record_failure | **GAP** |
| 8 | advisor budget-exhausted | SUCCESS (BUDGET_EXHAUSTED) | (none — deliberate skip) | **GAP** |
| 9 | advisor arg-error | ARG_ERROR | record_failure | **GAP** |
| 10 | advisor deadline (COMPOSE_TIMEOUT) | (see code ~2900) | per code | **GAP** |
| 11 | session-aware dispatch | via `_dispatch_session_aware_tool` | per outcome | `test_compose_loop_interpretation_review_dispatch.py` |
| 12 | generic ToolArgumentError | ARG_ERROR | record_failure | `fake_llm_tool_argument_error_on_second` |
| 13 | narrow re-raise (AssertionError/Memory/Recursion/System) | PLUGIN_CRASH then raise | (none) | `fake_llm_assertion_error_on_second`, `test_compose_loop_audit_wiring.py` B2 |
| 14 | general Exception → plugin_crash break | PLUGIN_CRASH | (none) | `fake_llm_runtime_error_on_second`, audit-wiring B3 |
| 15 | success (mutation) | SUCCESS | record_success | `fake_llm_one_set_pipeline_tool_call` |
| 16 | success (discovery) | SUCCESS | (none — observation) | `fake_llm_two_tool_calls` |
| 17 | success + `get_plugin_schema` schema-loaded mark | SUCCESS | (none) | **GAP (explicit assertion on schemas_loaded)** |

The verbatim move (Phase 2) is correct only for the arms a test pins. Tasks 1–4 below close the GAP rows. Arms already covered get an *explicit status assertion* added only if the existing test doesn't already assert the `ComposerToolStatus` (verify per Task 5).

### Task 1: Characterization harness + the three argument-error arms (#1, #2, #3)

**Files:**
- Create: `tests/unit/web/composer/test_dispatch_arms_characterization.py`

- [ ] **Step 1: Write the failing test for arm #1 (JSON-decode failure)**

```python
"""Characterization tests pinning every terminal arm of _dispatch_tool_batch.

These tests assert the audit-envelope status (ComposerToolStatus), the
anti-anchor side-effect, and the LLM tool-message shape for each arm. They
exist to make the Phase-2 verbatim extraction of the dispatch loop provably
behaviour-preserving for the audit trail — a dropped or reordered
recorder.record(finish_*) on any arm must turn one of these RED.

Driver: ComposerServiceImpl._run_one_turn_for_test(llm=...). The returned
ComposeLoopTestResult exposes .tool_invocations (the recorder buffer) and
.tool_outcomes.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from elspeth.contracts.composer_audit import ComposerToolStatus
from elspeth.web.composer.service import ComposerServiceImpl

from .conftest import _FakeComposeLLM, _fake_llm_response


def _raw_tool_call_llm(*, name: str, raw_arguments: str) -> _FakeComposeLLM:
    """LLM whose first turn emits ONE tool call with a raw (already-encoded)
    arguments string, bypassing _fake_llm_response's json.dumps. Used to inject
    malformed JSON / non-object payloads the decode arms must reject."""
    from .conftest import _FakeChoice, _FakeFunction, _FakeLLMResponse, _FakeMessage, _FakeToolCall

    first = _FakeLLMResponse(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[_FakeToolCall(id="call_raw", function=_FakeFunction(name=name, arguments=raw_arguments))],
                )
            )
        ]
    )
    return _FakeComposeLLM((first, _fake_llm_response(content="Done.")))


@pytest.mark.asyncio
async def test_arm_json_decode_failure_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    llm = _raw_tool_call_llm(name="get_pipeline_state", raw_arguments="{not valid json")
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses
    # The arm appends a role=tool message carrying an {"error": ...} payload.
    assert any(o.error_class is not None for o in result.tool_outcomes)
```

- [ ] **Step 2: Run it to verify it PASSES against current code**

Run:
```bash
.venv/bin/python -m pytest "tests/unit/web/composer/test_dispatch_arms_characterization.py::test_arm_json_decode_failure_records_arg_error" -v
```
Expected: PASS. (Characterization tests pin *existing* behaviour, so they pass now; their job is to FAIL if Phase 2 regresses the arm. If it ERRORS on a harness detail — fixture name, attribute — fix the test until it passes against current `service.py`.)

- [ ] **Step 3: Add arm #2 (non-dict arguments) and arm #3 (canonicalization failure)**

```python
@pytest.mark.asyncio
async def test_arm_non_dict_arguments_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    # Valid JSON, but a list rather than an object.
    llm = _raw_tool_call_llm(name="get_pipeline_state", raw_arguments=json.dumps([1, 2, 3]))
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses
    assert any(o.error_class in {"TypeError", "MissingRequiredPaths"} or o.error_class is not None
               for o in result.tool_outcomes)


@pytest.mark.asyncio
async def test_arm_required_paths_missing_records_arg_error(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    # set_source declares required paths; an empty object omits them.
    llm = _raw_tool_call_llm(name="set_source", raw_arguments=json.dumps({}))
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.ARG_ERROR in statuses
    assert any(o.error_class == "MissingRequiredPaths" for o in result.tool_outcomes)
```

- [ ] **Step 4: Run all three; verify PASS**

Run:
```bash
.venv/bin/python -m pytest "tests/unit/web/composer/test_dispatch_arms_characterization.py" -v
```
Expected: 3 passed. If arm #3's `error_class` differs from `MissingRequiredPaths` (e.g. the chosen tool routes through canonicalization first), adjust the assertion to the actual class observed — the goal is to PIN the real current behaviour, not impose an expected one.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/web/composer/test_dispatch_arms_characterization.py
git commit --no-verify -m "test(composer): characterize dispatch arg-error arms (#1-#3,#5)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 2: Discovery cache-hit arm (#4)

**Files:**
- Modify: `tests/unit/web/composer/test_dispatch_arms_characterization.py`

- [ ] **Step 1: Write the test — same cacheable discovery tool called twice in one turn**

```python
@pytest.mark.asyncio
async def test_arm_discovery_cache_hit_records_success_without_recompute(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    from .conftest import _FakeChoice, _FakeFunction, _FakeLLMResponse, _FakeMessage, _FakeToolCall

    # Two identical calls to a cacheable discovery tool in ONE assistant turn.
    # The second must be served from discovery_cache (cache_hit=True), recorded
    # SUCCESS, and must NOT break the §7.7 anchor.
    def _call(idx: int) -> _FakeToolCall:
        return _FakeToolCall(id=f"c{idx}", function=_FakeFunction(name="list_sources", arguments="{}"))

    first = _FakeLLMResponse(choices=[_FakeChoice(message=_FakeMessage(content=None, tool_calls=[_call(0), _call(1)]))])
    llm = _FakeComposeLLM((first, _fake_llm_response(content="Done.")))

    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    successes = [inv for inv in result.tool_invocations if inv.status == ComposerToolStatus.SUCCESS]
    assert len(successes) >= 2
    # At least one invocation recorded the cache_hit marker.
    assert any(getattr(inv, "cache_hit", False) for inv in result.tool_invocations)
```

- [ ] **Step 2: Run; verify PASS (or adjust to the real cacheable tool name)**

Run:
```bash
.venv/bin/python -m pytest "tests/unit/web/composer/test_dispatch_arms_characterization.py::test_arm_discovery_cache_hit_records_success_without_recompute" -v
```
Expected: PASS. If `list_sources` is not in `_CACHEABLE_DISCOVERY_TOOL_NAMES`, replace it with a tool that is — check:
```bash
.venv/bin/python -c "from elspeth.web.composer.tools._registry import _CACHEABLE_DISCOVERY_TOOL_NAMES as n; print(sorted(n))"
```
Use the first name printed; update the test accordingly.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/web/composer/test_dispatch_arms_characterization.py
git commit --no-verify -m "test(composer): characterize discovery cache-hit arm (#4)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 3: Advisor arms (#7 disabled, #8 budget-exhausted, #9 arg-error, #10 deadline)

**Files:**
- Modify: `tests/unit/web/composer/test_dispatch_arms_characterization.py`

- [ ] **Step 1: Write arm #7 (advisor disabled) — the anti-anchor `record_failure` side-effect must fire**

```python
@pytest.mark.asyncio
async def test_arm_advisor_disabled_records_success_payload_and_anchor_failure(
    composer_service_with_real_sessions: ComposerServiceImpl,
    result_session_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # composer_advisor_enabled defaults False in the test settings; assert the
    # disabled arm closes the envelope SUCCESS with an error payload and counts
    # the anti-anchor failure (per the code comment at service.py ~2811).
    svc = composer_service_with_real_sessions
    assert svc._settings.composer_advisor_enabled is False
    llm = _raw_tool_call_llm(name="request_advisor_hint", raw_arguments=json.dumps({"problem_summary": "x", "attempted_actions": []}))
    result = await svc._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    statuses = [inv.status for inv in result.tool_invocations]
    assert ComposerToolStatus.SUCCESS in statuses
    assert any(isinstance(o.response, dict) and "error" in o.response for o in result.tool_outcomes)
```

- [ ] **Step 2: Run; verify PASS**

Run:
```bash
.venv/bin/python -m pytest "tests/unit/web/composer/test_dispatch_arms_characterization.py::test_arm_advisor_disabled_records_success_payload_and_anchor_failure" -v
```
Expected: PASS. If the advisor-arg validation rejects `{"problem_summary","attempted_actions"}` before the disabled check, inspect `_validate_advisor_arguments` and supply the minimal valid shape; the disabled-deployment branch fires before the budget/arg checks per `service.py` ordering, so a minimally-parseable object suffices.

- [ ] **Step 3: Write arms #8/#9/#10 with `composer_advisor_enabled=True`**

```python
def _advisor_enabled_service(tmp_path: Any) -> ComposerServiceImpl:
    from .conftest import _make_settings, _mock_catalog, build_test_sessions_service
    settings = _make_settings(
        tmp_path,
        composer_advisor_enabled=True,
        composer_advisor_max_calls_per_compose=0,  # forces BUDGET_EXHAUSTED on first call
    )
    return ComposerServiceImpl(
        catalog=_mock_catalog(),
        settings=settings,
        sessions_service=build_test_sessions_service(data_dir=tmp_path),
    )


@pytest.mark.asyncio
async def test_arm_advisor_budget_exhausted_records_success_no_anchor(tmp_path: Any) -> None:
    from uuid import uuid4
    from datetime import UTC, datetime
    from elspeth.web.sessions.models import sessions_table

    svc = _advisor_enabled_service(tmp_path)
    session_id = str(uuid4())
    now = datetime.now(UTC)
    with svc._sessions_service._engine.begin() as conn:
        conn.execute(sessions_table.insert().values(
            id=session_id, user_id="u", auth_provider_type="local", title="t",
            trust_mode="auto_commit", density_default="high", created_at=now, updated_at=now))

    llm = _raw_tool_call_llm(name="request_advisor_hint",
                             raw_arguments=json.dumps({"problem_summary": "x", "attempted_actions": []}))
    result = await svc._run_one_turn_for_test(llm=llm, session_id=session_id)

    # BUDGET_EXHAUSTED closes SUCCESS and DELIBERATELY does not touch anti_anchor
    # (service.py ~2848: budget exhaustion is an operator-policy signal, not an
    # LLM-repetition pattern). Pin the SUCCESS + payload status.
    assert ComposerToolStatus.SUCCESS in [inv.status for inv in result.tool_invocations]
    assert any(isinstance(o.response, dict) and o.response.get("status") == "BUDGET_EXHAUSTED"
               for o in result.tool_outcomes)
```

- [ ] **Step 4: Run arms #7/#8; verify PASS**

Run:
```bash
.venv/bin/python -m pytest "tests/unit/web/composer/test_dispatch_arms_characterization.py" -k advisor -v
```
Expected: PASS. Arms #9 (arg-error: pass `attempted_actions` as a non-list) and #10 (deadline: construct the service with `composer_timeout_seconds` near-zero so `deadline - loop.time() <= 0`) follow the same shape — add them, run, pin the actually-observed status/payload. If arm #10 is impractical to trigger deterministically via the driver (the deadline is computed inside `_run_one_turn_for_test`), document that gap explicitly in a module-level comment naming arm #10 as covered only by the live timeout path, rather than silently dropping it.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/web/composer/test_dispatch_arms_characterization.py
git commit --no-verify -m "test(composer): characterize advisor arms (#7-#10)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 4: `get_plugin_schema` schema-loaded mark (#17) + coverage audit of pre-covered arms

**Files:**
- Modify: `tests/unit/web/composer/test_dispatch_arms_characterization.py`

- [ ] **Step 1: Write arm #17 — successful `get_plugin_schema` marks the (type,name) pair loaded**

```python
@pytest.mark.asyncio
async def test_arm_get_plugin_schema_marks_schema_loaded(
    fake_composer_service: ComposerServiceImpl,
    result_session_id: str,
) -> None:
    llm = _raw_tool_call_llm(
        name="get_plugin_schema",
        raw_arguments=json.dumps({"plugin_type": "source", "name": "csv"}),
    )
    result = await fake_composer_service._run_one_turn_for_test(llm=llm, session_id=result_session_id)

    assert ComposerToolStatus.SUCCESS in [inv.status for inv in result.tool_invocations]
    loaded = fake_composer_service._schemas_loaded_for_session(result_session_id)
    assert ("source", "csv") in loaded
```

- [ ] **Step 2: Run; verify PASS**

Run:
```bash
.venv/bin/python -m pytest "tests/unit/web/composer/test_dispatch_arms_characterization.py::test_arm_get_plugin_schema_marks_schema_loaded" -v
```
Expected: PASS. (`_schemas_loaded_for_session` is `service.py:4191`; the mock catalog's `get_schema` returns a `csv` source schema per conftest.)

- [ ] **Step 3: Audit pre-covered arms (#6, #11, #12, #13, #14, #15, #16) for an explicit status assertion**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_compose_loop_audit_wiring.py \
  tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py -v
```
For each arm in the inventory table marked with an existing test, confirm that test asserts the `ComposerToolStatus` (or the `ComposerPluginCrashError.tool_invocations` sequence for #13/#14). If an arm's existing test asserts only the *outcome* and not the audit status, add a one-line status assertion to `test_dispatch_arms_characterization.py` referencing the same fixture. Do NOT duplicate a fully-covered arm.

- [ ] **Step 4: Run the full characterization file + the existing audit suite together**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_dispatch_arms_characterization.py \
  tests/unit/web/composer/test_compose_loop_audit_wiring.py -q
```
Expected: all pass. Every row of the inventory table is now pinned by a named test (or explicitly documented as live-only for #10).

- [ ] **Step 5: Commit**

```bash
git add tests/unit/web/composer/test_dispatch_arms_characterization.py
git commit --no-verify -m "test(composer): characterize schema-loaded arm (#17) + audit pre-covered arms

Every terminal arm of _dispatch_tool_batch now pinned by status assertions.
Gates the Phase-2 verbatim extraction.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — Verbatim extraction of `_dispatch_tool_batch`

Move the loop body to `tool_batch.py` with all arms intact. NO idiom collapse, NO logic change. The characterization suite (Phase 1) + full composer suite must stay green at the single commit that lands this.

### Task 5: Define `ToolBatchContext` and `BatchAccumulator`

**Files:**
- Create: `src/elspeth/web/composer/tool_batch.py`

- [ ] **Step 1: Create the module with the two carriers (new code, fully specified)**

```python
"""Per-tool-call dispatch pipeline for the composer compose loop.

Extracted verbatim from ComposerServiceImpl._dispatch_tool_batch (service.py)
to take the single largest method out of the god class. The loop body is
UNCHANGED; only its enclosing context is made explicit via the two carriers
below, replacing the prior nested-closure capture of loop-invariant inputs and
loop-carried accumulators.

Behaviour-preservation contract: every terminal arm's
recorder.record(finish_*) / anti_anchor.record_* / llm_messages.append /
budget-class side-effect is identical to the pre-extraction method. Pinned by
tests/unit/web/composer/test_dispatch_arms_characterization.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from elspeth.web.composer.service import ComposerServiceImpl
    # ... (carrier/result types imported from their existing homes; fill in
    # during Step 2 from service.py's current imports: _CallModelOutcome,
    # _DispatchOutcome, _ToolOutcome, _CachedDiscoveryPayload,
    # _RuntimePreflightCache, AntiAnchorTracker, BufferingRecorder,
    # CompositionState, ValidationSummary, ValidationResult,
    # ComposerProgressSink, etc.)


@dataclass(frozen=True, slots=True)
class ToolBatchContext:
    """Loop-invariant inputs to the dispatch loop, built once per batch.

    Mirrors the keyword parameters of the former _dispatch_tool_batch plus the
    `self`-derived handles the loop reads. `service` is the bound
    ComposerServiceImpl so the loop can still call the methods that remain on
    the class (_call_advisor_with_audit, _dispatch_session_aware_tool,
    _cached_runtime_preflight, _validate_advisor_arguments,
    _mark_plugin_schema_loaded). This is a deliberate collaborator handle, not
    a god-object: ToolBatchContext carries no behaviour.
    """

    service: "ComposerServiceImpl"
    recorder: Any            # BufferingRecorder
    anti_anchor: Any         # AntiAnchorTracker
    discovery_cache: dict[str, Any]
    runtime_preflight_cache: Any
    session_id: str | None
    user_id: str | None
    user_message_id: str | None
    user_message_content: str | None
    current_state_id: str | None
    actor: str
    initial_version: int
    deadline: float
    progress: Any            # ComposerProgressSink | None
    session_scope: str
    turn_sessions_service: Any
    turn_session_uuid: Any   # UUID | None
    turn_preferences: Any


@dataclass(slots=True)
class BatchAccumulator:
    """Loop-carried state that rebinds per iteration.

    Honest mutable counterpart to ToolBatchContext: every name here is rebound
    or appended inside the loop. Initialised from the batch entry state.
    """

    state: Any               # CompositionState (rebinds)
    last_validation: Any     # ValidationSummary | None (rebinds)
    last_runtime_preflight: Any  # ValidationResult | None (rebinds)
    advisor_calls_used: int  # rebinds; driver owns it across iterations
    turn_has_mutation: bool = False
    turn_has_discovery: bool = False
    all_cache_hits: bool = True
    proposals_this_turn: int = 0
    mutation_success_observed: bool = False
    plugin_crash: Any = None
    plugin_crash_cause: Any = None
    tool_outcomes: list[Any] = field(default_factory=list)
    decoded_args_by_call_id: dict[str, dict[str, Any]] = field(default_factory=dict)
```

- [ ] **Step 2: Fill the real imports**

Open `service.py`, copy the exact import statements for the types referenced above (search for `_DispatchOutcome`, `_ToolOutcome`, `_CachedDiscoveryPayload`, `AntiAnchorTracker`, `BufferingRecorder`, `begin_dispatch`, `begin_dispatch_or_arg_error`, `dispatch_with_audit`, `finish_*`, `execute_tool`, `run_sync_in_worker`, `emit_progress`, `_make_cache_key`, `_TOOL_REQUIRED_PATHS`, `_find_missing_required_paths`, the predicate helpers `is_discovery_tool`/`is_mutation_tool`/`is_session_aware_tool`/`is_cacheable_discovery_tool`/`is_blob_store_only_mutation_tool`, `_serialize_tool_result`, `_arg_error_payload`, `_result_from_cached_discovery_payload`, `_cached_discovery_payload`, `build_tool_proposal_summary`, `safe_response_model`, `_MAX_PENDING_PROPOSALS_PER_TURN`) into `tool_batch.py`. Resolve each to its real module.

- [ ] **Step 3: Type-check the new module compiles (no `run_tool_batch` yet)**

Run:
```bash
.venv/bin/python -m mypy src/elspeth/web/composer/tool_batch.py
```
Expected: clean (only the two dataclasses exist so far).

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/tool_batch.py
git commit --no-verify -m "refactor(composer): add ToolBatchContext/BatchAccumulator carriers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 6: Move the loop body into `run_tool_batch`

**Files:**
- Modify: `src/elspeth/web/composer/tool_batch.py`
- Modify: `src/elspeth/web/composer/service.py` (`_dispatch_tool_batch` body → thin delegation)

- [ ] **Step 1: Add `run_tool_batch` and paste the loop body verbatim**

Add to `tool_batch.py`:
```python
async def run_tool_batch(
    *,
    call_model: Any,        # _CallModelOutcome
    ctx: ToolBatchContext,
    acc: BatchAccumulator,
    llm_messages: list[dict[str, Any]],
) -> tuple[Any, int]:       # (_DispatchOutcome, advisor_calls_used)
    """Phase P3 of the compose loop — dispatch one tool batch.

    Body is the verbatim former ComposerServiceImpl._dispatch_tool_batch loop,
    with mechanical substitutions only:
      - `self.X`            -> `ctx.service.X`
      - the loop-invariant locals (recorder, anti_anchor, discovery_cache,
        runtime_preflight_cache, session_id, user_id, user_message_id,
        user_message_content, current_state_id, actor, initial_version,
        deadline, progress, session_scope, turn_sessions_service,
        turn_session_uuid, turn_preferences) -> `ctx.<name>`
      - the loop-carried locals (state, last_validation,
        last_runtime_preflight, advisor_calls_used, turn_has_mutation,
        turn_has_discovery, all_cache_hits, proposals_this_turn,
        mutation_success_observed, plugin_crash, plugin_crash_cause,
        tool_outcomes, decoded_args_by_call_id) -> `acc.<name>`
      - `self._validate_advisor_arguments` etc. -> `ctx.service.<method>`
    NO other change. The `_append_tool_outcome` closure and the per-iteration
    default-arg closures (`_do_dispatch`, `_make_preflight_callback`,
    `_arg_error_payload_factory`, `_version_after`) are pasted unchanged; they
    capture loop-locals exactly as before.
    """
    # <PASTE: the entire body of the former _dispatch_tool_batch from the
    #  `assistant_message = call_model.assistant_message` line through the
    #  `return dispatch, advisor_calls_used` line, with the substitutions above.>
```

Mechanical guidance for the paste:
- The advisor escape-hatch arm calls `self._call_advisor_with_audit(...)` and reads `advisor_calls_used` — rebind via `acc.advisor_calls_used` and `ctx.service._call_advisor_with_audit`.
- The final `_DispatchOutcome(...)` construction reads every accumulator — source each field from `acc.*`.
- `self._phase3_last_expected_current_state_id = pre_state_id` becomes `ctx.service._phase3_last_expected_current_state_id = pre_state_id` (preserve this — it is read by the persistence step).

- [ ] **Step 2: Replace `_dispatch_tool_batch`'s body with delegation**

In `service.py`, keep the method signature identical but make the body build the carriers and delegate:
```python
    async def _dispatch_tool_batch(self, *, call_model, state, last_validation,
        last_runtime_preflight, llm_messages, recorder, anti_anchor,
        discovery_cache, runtime_preflight_cache, session_id, user_id,
        user_message_id, user_message_content, current_state_id, actor,
        initial_version, deadline, progress, session_scope, advisor_calls_used):
        from elspeth.web.composer.tool_batch import (
            BatchAccumulator, ToolBatchContext, run_tool_batch,
        )
        turn_sessions_service = self._require_sessions_service() if session_id is not None else None
        turn_session_uuid = UUID(session_id) if session_id is not None else None
        turn_preferences = (
            await turn_sessions_service.get_composer_preferences(turn_session_uuid)
            if turn_sessions_service is not None and turn_session_uuid is not None
            else None
        )
        ctx = ToolBatchContext(
            service=self, recorder=recorder, anti_anchor=anti_anchor,
            discovery_cache=discovery_cache, runtime_preflight_cache=runtime_preflight_cache,
            session_id=session_id, user_id=user_id, user_message_id=user_message_id,
            user_message_content=user_message_content, current_state_id=current_state_id,
            actor=actor, initial_version=initial_version, deadline=deadline,
            progress=progress, session_scope=session_scope,
            turn_sessions_service=turn_sessions_service,
            turn_session_uuid=turn_session_uuid, turn_preferences=turn_preferences,
        )
        acc = BatchAccumulator(
            state=state, last_validation=last_validation,
            last_runtime_preflight=last_runtime_preflight,
            advisor_calls_used=advisor_calls_used,
        )
        return await run_tool_batch(call_model=call_model, ctx=ctx, acc=acc, llm_messages=llm_messages)
```
Note: the `turn_sessions_service`/`turn_session_uuid`/`turn_preferences` setup moves OUT of the loop body and into this delegation (it was loop-invariant in the original — lines ~2364-2370). Delete those three lines from the pasted body in `tool_batch.py` since they now arrive via `ctx`.

- [ ] **Step 3: Type-check both files**

Run:
```bash
.venv/bin/python -m mypy src/elspeth/web/composer/tool_batch.py src/elspeth/web/composer/service.py
```
Expected: clean. Fix any `Any`-vs-concrete signature mismatches by importing the real types in `tool_batch.py` (prefer real types over `Any` wherever the import is non-cyclic; `ComposerServiceImpl` stays under `TYPE_CHECKING`).

- [ ] **Step 4: Run the characterization suite + full composer suite**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/ tests/property/web/composer/ -q
```
Expected: same pass count as the Phase 0 baseline. ANY regression here is a botched substitution — bisect by arm using the characterization tests, do not paper over.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/tool_batch.py src/elspeth/web/composer/service.py
git commit --no-verify -m "refactor(composer): extract _dispatch_tool_batch to tool_batch.run_tool_batch (verbatim)

1,298-line method relocated unchanged; service.py delegates via
ToolBatchContext/BatchAccumulator. Behaviour pinned by dispatch-arm
characterization suite. No logic change.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Collapse the terminal-arm emit idiom

Only now — with the move green — collapse the repeated `recorder.record(finish_*) → _append_tool_outcome → anti_anchor.record_* → llm_messages.append` idiom, where each per-arm variation is reviewable in isolation.

### Task 7: Extract `tool_batch_arms.py` emit helpers

**Files:**
- Create: `src/elspeth/web/composer/tool_batch_arms.py`
- Modify: `src/elspeth/web/composer/tool_batch.py`

- [ ] **Step 1: Write a focused unit test for the emit helper's invariants**

```python
# tests/unit/web/composer/test_tool_batch_arms.py
"""Unit tests for the uniform terminal-arm emit helpers.

The helpers centralise the ordering invariant documented in the former
_dispatch_tool_batch docstring: record(finish_*) -> append outcome ->
anti_anchor (optional) -> llm_messages.append. These tests pin that the
optional anti_anchor skip is honoured per-arm (budget-exhaustion /
advisor-disabled arms must NOT call anti_anchor.record_failure).
"""
from __future__ import annotations
# ... build a fake recorder/anti_anchor/accumulator, call emit_arg_error and
# emit_budget_exhausted, assert anti_anchor.record_failure called exactly once
# for the former and zero times for the latter, and that llm_messages got one
# role=tool entry each.
```
Fill the body using `unittest.mock.MagicMock` doubles for `recorder`/`anti_anchor`; assert call counts.

- [ ] **Step 2: Run it to verify it FAILS (helpers don't exist yet)**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_tool_batch_arms.py -v
```
Expected: FAIL with `ImportError`/`AttributeError` for `emit_arg_error`.

- [ ] **Step 3: Implement the emit helpers, capturing the per-arm variations as explicit parameters**

```python
# src/elspeth/web/composer/tool_batch_arms.py
"""Uniform terminal-arm emit helpers for the dispatch loop.

Each helper performs the ordering-invariant sequence ONCE; the per-arm
variations that previously diverged inline are now explicit parameters:
  - touch_anti_anchor: whether to call anti_anchor.record_failure (budget /
    advisor-disabled arms pass False — they are policy signals, not LLM
    repetition).
  - budget_class: 'discovery' | 'mutation' | None — which turn flag to set.
"""
# def emit_arg_error(*, acc, ctx, audit, tool_call, error_class, error_message,
#                    error_payload, budget_class) -> None: ...
# def emit_success(...) -> None: ...
# def emit_budget_exhausted(...) -> None: ...   # touch_anti_anchor=False
# def emit_cache_hit(...) -> None: ...
```

- [ ] **Step 4: Run the arms unit test; verify PASS**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/test_tool_batch_arms.py -v
```
Expected: PASS.

- [ ] **Step 5: Rewrite ONE arm in `run_tool_batch` to use the helper, run characterization + full suite**

Convert the JSON-decode arm (#1) first. Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/ -q
```
Expected: green. Repeat per arm, ONE arm per commit, running the full composer suite each time so a divergence is bisectable to a single arm.

- [ ] **Step 6: Commit (one arm per commit)**

```bash
git add src/elspeth/web/composer/tool_batch_arms.py src/elspeth/web/composer/tool_batch.py tests/unit/web/composer/test_tool_batch_arms.py
git commit --no-verify -m "refactor(composer): collapse arg-error arm onto emit helper (#1)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```
Repeat Steps 5–6 for each remaining arm.

---

## Phase 4 — Sync ToolDeclaration convergence (optional, in-bounds)

### Task 8: Fold the `get_plugin_schema` post-success hook toward the declaration

**Files:**
- Modify: `src/elspeth/web/composer/tools/declarations.py` (add optional sync `post_success` hook field)
- Modify: `src/elspeth/web/composer/tools/sessions.py` or the plane declaring `get_plugin_schema`
- Modify: `src/elspeth/web/composer/tool_batch.py` (replace the `if tool_name == "get_plugin_schema"` branch with a declaration-driven hook call)

- [ ] **Step 1:** Add an optional `post_success: Callable | None = None` field to `ToolDeclaration` with a `__post_init__` note that async carve-outs remain deferred to `elspeth-f5da936747` (do NOT widen to async). Keep the arm #17 characterization test as the gate.
- [ ] **Step 2:** Move the schema-loaded mark into the `get_plugin_schema` declaration's `post_success` hook; replace the inline `if tool_name == "get_plugin_schema"` in `run_tool_batch` with `decl.post_success and decl.post_success(...)`.
- [ ] **Step 3:** Run `tests/unit/web/composer/ -q`; arm #17 test must stay green.
- [ ] **Step 4:** Commit `--no-verify`.

**Async carve-outs (`request_advisor_hint`, `request_interpretation_review`) stay as explicit branches — out of scope, deferred to `elspeth-f5da936747`.**

---

## Phase 5 — Remaining oversized-method extractions

Each task mirrors Phase 2's verbatim-move discipline: characterize-if-needed → move → green → commit. One method per task, each its own review checkpoint.

### Task 9: Extract `_persist_turn_audit` → `turn_audit.py`
- [ ] Confirm coverage in `test_compose_loop_persistence.py` / `test_compose_loop_audit_wiring.py`; add a status-pinning test if the persistence sequence isn't asserted.
- [ ] Move `_persist_turn_audit` (service.py ~2101-2256) to `turn_audit.py` as `persist_turn_audit(service, ...)`; service delegates.
- [ ] `pytest tests/unit/web/composer/ -q` green; mypy clean; commit `--no-verify`.

### Task 10: Extract `_compute_availability` + `ComposerAvailability` → `availability.py`
- [ ] Note: `ComposerAvailability` is imported by tests (`from elspeth.web.composer.service import ComposerAvailability`) and conftest monkeypatches `ComposerServiceImpl._compute_availability`. Re-export `ComposerAvailability` from `service.py` (`from elspeth.web.composer.availability import ComposerAvailability`) so existing imports keep working WITHOUT a shim — this is a real re-export of the moved symbol, not a compat alias.
- [ ] Move; keep the monkeypatch target (`ComposerServiceImpl._compute_availability`) a real method that delegates to the module function.
- [ ] `pytest tests/unit/web/composer/ -q` green; mypy clean; commit `--no-verify`.

### Task 11: Extract `_finalize_no_tool_response` → `no_tool_finalize.py`
- [ ] Move (service.py ~1439-1718) to `finalize_no_tool_response(service, ...)`; service delegates.
- [ ] `pytest tests/unit/web/composer/ -q` green; mypy clean; commit `--no-verify`.

---

## Phase 6 — Reconciliation & landing prep

### Task 12: Full surface verification + fingerprint reconciliation

- [ ] **Step 1: Full composer + adjacent suites**

Run:
```bash
.venv/bin/python -m pytest tests/unit/web/composer/ tests/property/web/composer/ \
  tests/integration/pipeline/test_composer_llm_eval_characterization.py -q
```
Expected: green, pass count >= Phase 0 baseline + the new characterization tests.

- [ ] **Step 2: Type + lint the changed surface**

Run:
```bash
.venv/bin/python -m mypy src/elspeth/web/composer/
.venv/bin/python -m ruff check src/elspeth/web/composer/
```
Expected: clean.

- [ ] **Step 3: Measure the win**

Run:
```bash
wc -l src/elspeth/web/composer/service.py src/elspeth/web/composer/tool_batch.py \
  src/elspeth/web/composer/tool_batch_arms.py src/elspeth/web/composer/turn_audit.py \
  src/elspeth/web/composer/availability.py src/elspeth/web/composer/no_tool_finalize.py
```
Expected: `service.py` materially smaller (target trajectory ~1,500–2,000); no new file approaching the old monster.

- [ ] **Step 4: Reconcile tier-model fingerprints ONCE (not per-commit)**

Run the check, inspect `stale_allowlist_entries`, and reconcile per the rotation discipline (do NOT run the rotate tool bare; `git diff` + re-run after rotating; watch for duplicate-key data loss):
```bash
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli \
  check --rules trust_tier.tier_model --root src/elspeth --format json | \
  .venv/bin/python -m json.tool | head -60
```
If the new modules introduce findings, prefer per-file-rule collapse over per-line fingerprints for the new files (high-churn). Co-land the allowlist update in one commit.

- [ ] **Step 5: Commit the reconciliation**

```bash
git add -A
git commit --no-verify -m "chore(composer): reconcile tier-model allowlist after decomposition

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Update the Filigree epic**

Comment on `elspeth-6c9972ccbf` recording that the sync convergence (Phase 4) landed and the async carve-outs remain under `elspeth-f5da936747`; the registry substrate was already merged. Do not close (per-cluster closure criteria).

---

## Self-Review notes (author)

- **Spec coverage:** every spec section maps to a phase — sequence (Phase 1→3), module seams (Phase 2,5), interface carriers (Task 5), sync-only convergence (Phase 4), operational guardrails (Phase 0 venv, Phase 6 fingerprints).
- **Async deferral:** enforced in Task 8 and the inventory (#11 left as a branch) — does not re-open `elspeth-f5da936747`.
- **No silent truncation:** arm #10 (advisor deadline) is flagged as possibly live-only with an explicit documented gap rather than dropped.
- **Type consistency:** carrier field names in Task 5 are reused verbatim in Task 6's delegation and the `acc.*`/`ctx.*` substitution guidance.
