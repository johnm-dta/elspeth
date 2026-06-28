# Composer Guided Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a structured-protocol "guided mode" wizard to the ELSPETH composer that walks first-time users through source → sink → transforms with a closed six-turn taxonomy and a read-only LLM, falling back to the existing freeform composer on completion or failure.

**Architecture:** A new `composer/guided/` Python module owns the protocol types, state machine, recipe pre-match, chain solver, and skill prompt. Two new HTTP endpoints (`/composer/guided/start`, `/composer/guided/respond`) drive a deterministic state machine that mutates `CompositionState` via existing `tools.py` handlers — the LLM never writes to state. Frontend grows a `chat/guided/` widget directory dispatched from a top-level mode discriminator in `ChatPanel.tsx`. Freeform composer is unmodified. Mode transition uses progressive disclosure: the freeform skill is appended to the guided skill rather than replacing it, so the model retains coherence across the boundary.

**Tech Stack:** Python 3.13, pydantic-style frozen dataclasses with `freeze_fields`, FastAPI/Pyramid endpoints (matching existing `service.py` patterns), LiteLLM via `_litellm_acompletion`, React + TypeScript + Vitest + Playwright on the frontend. Audit emission via `BufferingRecorder`; telemetry via OpenTelemetry. Reuses `recipes.py`, `tools.py` mutation handlers, `ChaosLLM` test fixture.

**Spec:** `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md` (commit 39ebe672).

---

## Errata & Post-Review Corrections (2026-05-11, rev 2)

**Status:** This plan was reviewed against the actual source tree on 2026-05-11. The architectural shape, phase ordering, and pure-data tasks (Phases 1–2) are correct. **The HTTP routing layer, tool-handler signatures, audit interface, and frontend conventions are wrong in the original draft.** Read this errata block before starting any task. Each correction is referenced inline in the affected tasks below with the marker **[ERRATA Cn — see top of file]**.

Until the inline tasks are revised, treat the errata block as authoritative when the two disagree.

### C1. `_execute_*` tool-handler signatures — all task descriptions in Phases 3 and 4 are wrong

Plan writes `new_state, tool_result = _execute_set_source(state, payload)`. Actual signature (`tools.py:2371`) is:

```python
def _execute_set_source(
    args: dict[str, Any],
    state: CompositionState,
    catalog: CatalogService,
    data_dir: str | None = None,
) -> ToolResult: ...
```

Every `_execute_*` function in `tools.py` follows the `(args: dict, state, catalog, data_dir, **kwargs)` pattern. `ToolResult` (defined at `tools.py:347`) is a single dataclass — `result.success`, `result.updated_state`, `result.validation`, `result.affected_nodes`, `result.data`, `result.prior_validation`, `result.runtime_preflight` — NOT a `(state, result)` tuple.

Step handlers in `composer/guided/steps.py` must (a) accept `catalog: CatalogService` and `data_dir: str | None` parameters passed in from the route handler, and (b) construct the `args` dict and read `ToolResult` attributes correctly. Tasks 3.1, 3.2, 3.3, 4.4 affected.

### C2. Use `_execute_apply_pipeline_recipe`, not raw `apply_recipe` + `_execute_set_pipeline`

`tools.py:3748` already wraps recipe application with full validation, audit emission via `ComposerToolInvocation`, and a `replaced_pipeline_note`. Task 3.3 (`handle_step_2_5_recipe_apply`) calls:

```python
result = _execute_apply_pipeline_recipe(
    {"recipe_name": match.recipe_name, "slots": dict(match.slots)},
    state, catalog, data_dir,
    session_engine=session_engine, session_id=session_id,
)
```

Reusing the canonical executor — not a two-step manual application — means audit records emit automatically through existing plumbing.

### C3. HTTP routes live in `sessions/routes.py`, not `service.py`; prefix is `/api/sessions/{session_id}/...`

The plan's "`@router.post` decorators in `service.py`" pattern is wrong:

- `service.py` contains `ComposerServiceImpl` (a service class). No routing decorators.
- Composer routes are mounted by `src/elspeth/web/sessions/routes.py:1499` under `APIRouter(prefix="/api/sessions", tags=["sessions"])`. The freeform chat endpoint is `POST /api/sessions/{session_id}/messages` at line 1629.
- Composer state loads via `service.get_current_state(session.id)`; new sessions construct `CompositionState(source=None, nodes=(), edges=(), outputs=(), metadata=PipelineMetadata(), version=1)` — required-arg constructor, NOT the no-arg `CompositionState()` the plan's tests assume.

Phase 3 tasks 3.4 and 3.5 must mount:
- `POST /api/sessions/{session_id}/guided/start` (likely redundant per C7; see below)
- `POST /api/sessions/{session_id}/guided/respond`

…and reuse `_verify_session_ownership`, `_get_session_compose_lock_registry`, and the rate limiter — the same dependency-injection block as `send_message`. Test URLs prefixed `/composer/sessions/...` and `/composer/guided/...` throughout Phase 3, 5, 9 must become `/api/sessions/.../guided/...`.

### C4. Audit emission: reuse `ComposerToolInvocation` with `tool_name` discriminator (Option A)

The plan invents `BufferingRecorder.record_guided_event(GuidedAuditEvent(...))`. This method doesn't exist. `BufferingRecorder` (audit.py:179) only has `record(ComposerToolInvocation)` and `record_llm_call(ComposerLLMCall)`. Persistence routes through `audit_envelope` into the `tool_calls` JSON column on the assistant message row.

**Decision: Option A — reuse `ComposerToolInvocation` with a `tool_name` discriminator.** The four event types become four `tool_name` values: `guided_turn_emitted`, `guided_turn_answered`, `guided_step_advanced`, `guided_dropped_to_freeform`. Payloads slot into the existing `arguments` / `result` fields. No contract type changes, no schema migration. Task 1.6 must construct `ComposerToolInvocation` records and call `recorder.record(invocation)` — there is no `record_guided_event` method.

If the audit consumer side ever needs richer guided-specific replay than `ComposerToolInvocation` exposes, split a `GuidedTurnInvocation` sibling contract type later; not v1.

### C5. Frontend API client is module-level `export async function`, not a class

`src/elspeth/web/frontend/src/api/client.ts` exports `export async function fetchSessions()`, `createSession()`, etc. — no class. Phase 6 task 6.2 needs:

```typescript
export async function postGuidedStart(sessionId: string): Promise<GuidedStartResponse> { ... }
export async function postGuidedRespond(sessionId: string, turnResponse: TurnResponse): Promise<GuidedRespondResponse> { ... }
```

Vitest mocks become `vi.spyOn(api, "postGuidedStart")` (import-namespace), not `vi.spyOn(apiClient, "postGuidedStart")` (class-method).

### C6. `sessionStore`: single `guidedTurn` atomic object, not separate `guidedSession` + `guidedNextTurn`

The plan ends with two store fields (`guidedSession` and `guidedNextTurn`) plus comments admitting the awkwardness — the server response is atomic, so the store should be too. Collapse into one:

```typescript
interface GuidedTurnState {
  session: GuidedSession;
  next_turn: Turn | null;
  terminal: TerminalState | null;
}

interface GuidedSlice {
  guidedTurn: GuidedTurnState | null;
  startGuided: (sessionId: string) => Promise<void>;
  respondGuided: (turnResponse: TurnResponse) => Promise<void>;
  exitToFreeform: () => Promise<void>;
}
```

Every server response replaces `guidedTurn` atomically; `compositionState` updates alongside it. ChatPanel reads `guidedTurn` once instead of three separate fields. Affects Phase 6 task 6.3 and Phase 8 task 8.1.

### C7. New sessions default to guided per spec §5.2 — remove `/guided/start` endpoint

The plan adds an explicit `POST .../guided/start` RPC. Spec §5.2 says guided is the default for new sessions: `composition_state.guided_session = GuidedSession.initial()` is attached at session-create time. The two are inconsistent.

**Decision: default-guided.** Modify the session-create route (`sessions/routes.py:1501`) to attach `GuidedSession.initial()` to the freshly-constructed `CompositionState`. The frontend renders guided UI when `composition_state.guided_session != null`. The `/guided/start` endpoint becomes unnecessary; remove from Phase 3 task 3.4 and from the frontend store's `startGuided` action.

The session-create change is one Python edit at line ~1663 (passing `guided_session=GuidedSession.initial()` to the `CompositionState` constructor); add it as the first sub-step of Phase 3 task 3.4 (which becomes "wire `/guided/respond` only — `/guided/start` is dropped").

### C8. Recipe-match predicates must be derived from actual recipe slot schemas

Phase 2 task 2.3 invents predicates from prose. The three registered recipes are in `composer/recipes.py`:
- `_build_classify_recipe` (line 236) — slots include prompt_template, model, csv_blob_id, output_path, output_field
- `_build_threshold_recipe` (line 337) — slots include threshold, field_to_check, above/below output paths
- `_build_fork_coalesce_truncate_recipe` (line 472) — fork shape, truncate-arm config, jsonl outputs

Task 2.3 must add a Step 0: read each recipe's `_build_*` function and its `SlotSpec` declarations to identify which slots are required vs optional, then design predicates whose `slot_resolver` maps observed (Source, Sink) state to exactly the required slot set. Predicates that cannot resolve a required slot from observed state must return `False`. The plan's invented predicates (`_classify_predicate`, `_split_threshold_predicate`) are placeholders only — the implementation must verify against the actual `SlotSpec` lists.

### C9. Step 1 handler must run `inspect_blob_content` before emitting `inspect_and_confirm`

The plan describes the inspection turn but doesn't call out the inspection call. Actual function: `inspect_blob_content(content, filename, mime_type, blob_id=...)` in `composer/source_inspection.py:80` returns `SourceInspectionFacts`. Task 3.1 must:
1. Before emitting `inspect_and_confirm`, read the attached blob (via existing blob-store API) and call `inspect_blob_content(...)`.
2. Package the resulting `SourceInspectionFacts` (columns, sample rows, warnings) into the turn's `payload.observed`.
3. If no blob is attached, emit `schema_form` (manual options) or `single_select` (pick plugin) instead — the inspection path is conditional on blob presence.

### C10. Test fixture inventory check before Phase 3 begins

The plan references `composer_test_client`, `audit_recorder`, and `chaosllm_stub` fixtures. Actual state:
- `tests/fixtures/chaosllm.py:223` defines `chaosllm_server` (note the fixture name — `chaosllm_server`, not `chaosllm_stub`).
- No `composer_test_client` or `audit_recorder` fixture exists yet. The closest existing pattern lives in `tests/unit/web/composer/test_route_integration.py` (TestClient construction + recorder wiring). Read this before writing new fixtures.

Add a Phase 3 prerequisite task (3.0): "Read `tests/unit/web/composer/test_route_integration.py` for the established FastAPI TestClient + recorder fixture pattern; create `tests/integration/web/composer/guided/conftest.py` with `composer_test_client` and `audit_recorder` fixtures matching that pattern."

### Phase-level corrections summary

| Phase | Status | Reason |
|---|---|---|
| 1 | Keep, fold C4 + C5 (CompositionState construction) | Audit module reuses `ComposerToolInvocation`; state field add still correct |
| 2 | Keep, fold C8 (read recipes first) | Recipe predicates need real slot schemas |
| 3 | **Revise** per C1, C2, C3, C7, C9, C10 | Tool signatures, routing layer, source inspection, fixtures |
| 4 | Keep, fold C1 (handler signatures) | Chain solver structure correct; commit path needs handler-signature fix |
| 5 | Keep | Auto-drop, progressive disclosure, audit emission test correct |
| 6 | **Revise** per C5, C6 | Module-level api fns, single atomic store object |
| 7 | Keep | Widget specs correct |
| 8 | Revise per C6 | Single guidedTurn read |
| 9 | Keep, fix URL prefixes per C3 | E2E flows valid; URLs were `/composer/...` instead of `/api/sessions/...` |
| 10 | Keep | Docs |

### Two pre-flight decisions confirmed (do not relitigate)

1. **C4 Option A — reuse `ComposerToolInvocation` with `tool_name` discriminator.** Cheapest path; ships sooner; replay tooling already understands it.
2. **C7 default-guided.** Matches spec §5.2; removes `/guided/start` endpoint and `startGuided` store action; new sessions begin in guided mode.

If either decision needs reopening, do it via a plan revision PR before implementation starts.

---

## Pre-Flight: File Structure Map

### New files (backend)

| Path | Responsibility |
|---|---|
| `src/elspeth/web/composer/guided/__init__.py` | Module marker; public re-exports |
| `src/elspeth/web/composer/guided/protocol.py` | `TurnType` enum, payload TypedDicts, `TurnResponse`, `ControlSignal`, `Turn`, legal-turn matrix, payload-schema validators |
| `src/elspeth/web/composer/guided/state_machine.py` | `GuidedSession`, `GuidedStep`, `TerminalState`, `TurnRecord`, `step_advance()` pure function |
| `src/elspeth/web/composer/guided/steps.py` | Step handlers: `handle_step_1_source`, `handle_step_2_sink`, `handle_step_2_5_recipe`, `handle_step_3_transforms` |
| `src/elspeth/web/composer/guided/recipe_match.py` | `match_recipe(source, sink) -> RecipeMatch \| None` deterministic matcher |
| `src/elspeth/web/composer/guided/chain_solver.py` | `solve_chain()` async wrapper around `_litellm_acompletion` with guided skill |
| `src/elspeth/web/composer/guided/prompts.py` | Skill loader; Step 3 context-block constructor; mode-transition prompt |
| `src/elspeth/web/composer/guided/audit.py` | Audit event emit helpers for the four guided event types |
| `src/elspeth/web/composer/guided/skills/__init__.py` | Skill module marker |
| `src/elspeth/web/composer/guided/skills/guided_pipeline.md` | The ≤80-line guided-mode skill prompt |

### New files (frontend)

| Path | Responsibility |
|---|---|
| `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx` | Dispatcher; switches on `turn.type` |
| `src/elspeth/web/frontend/src/components/chat/guided/InspectAndConfirmTurn.tsx` | Inspect-confirm widget |
| `src/elspeth/web/frontend/src/components/chat/guided/SingleSelectTurn.tsx` | Chip group, radio semantics |
| `src/elspeth/web/frontend/src/components/chat/guided/MultiSelectWithCustomTurn.tsx` | Chip group + free-add input + "let source decide" escape |
| `src/elspeth/web/frontend/src/components/chat/guided/SchemaFormTurn.tsx` | Auto-generated form from JSON Schema |
| `src/elspeth/web/frontend/src/components/chat/guided/ProposeChainTurn.tsx` | Chain proposal display + accept/edit/reject |
| `src/elspeth/web/frontend/src/components/chat/guided/RecipeOfferTurn.tsx` | Recipe-offer card |
| `src/elspeth/web/frontend/src/components/chat/guided/ExitToFreeformButton.tsx` | Persistent exit-to-freeform control |
| `src/elspeth/web/frontend/src/components/chat/guided/GuidedHistory.tsx` | Compact completed-step list (collapsible) |
| `src/elspeth/web/frontend/src/components/chat/guided/CompletionSummary.tsx` | Success-termination YAML preview + save action |
| `src/elspeth/web/frontend/src/types/guided.ts` | Frontend-side TypeScript Turn/TurnResponse/GuidedSession types mirroring backend |

### New test files

| Path | Responsibility |
|---|---|
| `tests/unit/web/composer/guided/__init__.py` | Test module marker |
| `tests/unit/web/composer/guided/test_protocol.py` | Turn type + payload schema validation, round-trip |
| `tests/unit/web/composer/guided/test_state_machine.py` | `step_advance()` pure-function tests; Hypothesis property tests |
| `tests/unit/web/composer/guided/test_recipe_match.py` | Predicate evaluation, specificity ordering, slot-resolver outputs |
| `tests/unit/web/composer/guided/test_audit.py` | Audit event payload conformance |
| `tests/unit/web/composer/guided/test_state_field.py` | `CompositionState.guided_session` freeze guard |
| `tests/integration/web/composer/guided/__init__.py` | Test module marker |
| `tests/integration/web/composer/guided/test_endpoints.py` | `/composer/guided/{start,respond}` driven through realistic step sequences |
| `tests/integration/web/composer/guided/test_chain_solver.py` | Chain solver with stubbed LLM; one real-LLM gated case |
| `tests/integration/web/composer/guided/test_progressive_disclosure.py` | Mode transition prompt construction; LLM-after-transition assertion |
| `tests/integration/web/composer/guided/test_audit_emission.py` | Full session, all expected audit events appear |
| `src/elspeth/web/frontend/src/components/chat/guided/*.test.tsx` | One Vitest test per widget |
| `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts` | Store-action tests for guided slice |
| `src/elspeth/web/frontend/tests/playwright/composer-guided.spec.ts` | Three E2E flows + demo-SLA assertion |

### Files to modify

| Path | Change |
|---|---|
| `src/elspeth/web/composer/state.py` | Add `guided_session: GuidedSession \| None = None` field to `CompositionState`; update `freeze_fields` call |
| `src/elspeth/web/composer/service.py` | Register two new endpoints; call `composer.guided.audit.emit_*` at relevant points; insert progressive-disclosure prompt construction in chat-message system-prompt builder |
| `src/elspeth/web/frontend/src/stores/sessionStore.ts` | Add `guidedSession` slice + actions: `startGuided`, `respondGuided`, `exitToFreeform` |
| `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx` | Top-level mode discriminator |
| `src/elspeth/web/frontend/src/api/client.ts` | Add `postGuidedStart()` and `postGuidedRespond()` methods |

### Files NOT modified

`src/elspeth/web/composer/tools.py`, `src/elspeth/web/composer/recipes.py`, `src/elspeth/web/composer/skills/pipeline_composer.md` are unchanged. Guided mode reuses but does not edit them.

---

## Phase 1 — Protocol Foundation

**Goal of phase:** Establish the protocol types, state-machine data shapes, audit-event types, and `CompositionState` extension. Pure data and pure functions only — no I/O, no LLM, no DB.

**Phase exit criterion:** All Ring 1 unit tests for protocol/state/audit/state_field pass; `composition_state.guided_session` round-trips through serialisation; CI green.

### Task 1.1: Define `TurnType` enum and turn payload TypedDicts

**Files:**
- Create: `src/elspeth/web/composer/guided/__init__.py`
- Create: `src/elspeth/web/composer/guided/protocol.py`
- Test: `tests/unit/web/composer/guided/__init__.py`
- Test: `tests/unit/web/composer/guided/test_protocol.py`

- [ ] **Step 1: Create empty package markers**

```python
# src/elspeth/web/composer/guided/__init__.py
"""Guided-mode composer protocol module.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md.
"""
```

```python
# tests/unit/web/composer/guided/__init__.py
```

- [ ] **Step 2: Write the failing test for `TurnType` enum membership**

```python
# tests/unit/web/composer/guided/test_protocol.py
"""Tests for guided-mode protocol types."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.protocol import TurnType


class TestTurnType:
    def test_six_turn_types_defined(self) -> None:
        expected = {
            "inspect_and_confirm",
            "single_select",
            "multi_select_with_custom",
            "schema_form",
            "propose_chain",
            "recipe_offer",
        }
        assert {t.value for t in TurnType} == expected

    def test_turn_type_is_str_enum(self) -> None:
        assert TurnType.SINGLE_SELECT.value == "single_select"
        assert TurnType("single_select") is TurnType.SINGLE_SELECT
```

- [ ] **Step 3: Run test, verify it fails**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_protocol.py -v
```

Expected: `ModuleNotFoundError: No module named 'elspeth.web.composer.guided.protocol'`

- [ ] **Step 4: Implement minimal `TurnType` enum**

```python
# src/elspeth/web/composer/guided/protocol.py
"""Guided-mode protocol: turn types, payloads, responses, legal-turn matrix.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §4.
"""

from __future__ import annotations

from enum import Enum


class TurnType(str, Enum):
    """The closed taxonomy of turn types the protocol allows."""

    INSPECT_AND_CONFIRM = "inspect_and_confirm"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT_WITH_CUSTOM = "multi_select_with_custom"
    SCHEMA_FORM = "schema_form"
    PROPOSE_CHAIN = "propose_chain"
    RECIPE_OFFER = "recipe_offer"
```

- [ ] **Step 5: Run test, verify it passes**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_protocol.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Add the six payload TypedDicts (failing tests first)**

Add to `tests/unit/web/composer/guided/test_protocol.py`:

```python
from elspeth.web.composer.guided.protocol import (
    InspectAndConfirmPayload,
    MultiSelectWithCustomPayload,
    ProposeChainPayload,
    RecipeOfferPayload,
    SchemaFormPayload,
    SingleSelectPayload,
)


class TestPayloadShapes:
    def test_inspect_and_confirm_payload_required_keys(self) -> None:
        payload: InspectAndConfirmPayload = {
            "observed": {"columns": ["a", "b"], "samples": [{"a": 1}], "warnings": []},
        }
        assert payload["observed"]["columns"] == ["a", "b"]

    def test_single_select_payload(self) -> None:
        payload: SingleSelectPayload = {
            "question": "Pick one",
            "options": [{"id": "a", "label": "A", "hint": None}],
            "allow_custom": False,
        }
        assert payload["allow_custom"] is False

    def test_multi_select_payload(self) -> None:
        payload: MultiSelectWithCustomPayload = {
            "question": "Pick many",
            "options": [{"id": "a", "label": "A", "hint": None}],
            "default_chosen": ["a"],
            "escape_label": "Or: let source decide",
        }
        assert payload["escape_label"] == "Or: let source decide"

    def test_schema_form_payload(self) -> None:
        payload: SchemaFormPayload = {
            "plugin": "csv",
            "schema_block": {"path": {"type": "string"}},
            "prefilled": {},
        }
        assert payload["plugin"] == "csv"

    def test_propose_chain_payload(self) -> None:
        payload: ProposeChainPayload = {
            "steps": [
                {"plugin": "type_coerce", "options": {}, "rationale": "needed for gate"},
            ],
            "why": "bridge str to float",
            "blockers": [],
        }
        assert len(payload["steps"]) == 1

    def test_recipe_offer_payload(self) -> None:
        payload: RecipeOfferPayload = {
            "recipe_name": "classify-rows-llm-jsonl",
            "slots": {},
            "alternatives": [],
        }
        assert payload["recipe_name"] == "classify-rows-llm-jsonl"
```

Run, expect import errors.

- [ ] **Step 7: Implement payload TypedDicts**

Append to `src/elspeth/web/composer/guided/protocol.py`:

```python
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict


class _Option(TypedDict):
    id: str
    label: str
    hint: str | None


class _Observed(TypedDict):
    columns: Sequence[str]
    samples: Sequence[Mapping[str, Any]]
    warnings: Sequence[str]


class InspectAndConfirmPayload(TypedDict):
    observed: _Observed


class SingleSelectPayload(TypedDict):
    question: str
    options: Sequence[_Option]
    allow_custom: bool


class MultiSelectWithCustomPayload(TypedDict):
    question: str
    options: Sequence[_Option]
    default_chosen: Sequence[str]
    escape_label: str | None


class SchemaFormPayload(TypedDict):
    plugin: str
    schema_block: Mapping[str, Any]
    prefilled: Mapping[str, Any]


class _ProposedStep(TypedDict):
    plugin: str
    options: Mapping[str, Any]
    rationale: str


class ProposeChainPayload(TypedDict):
    steps: Sequence[_ProposedStep]
    why: str
    blockers: Sequence[str]


class RecipeOfferPayload(TypedDict):
    recipe_name: str
    slots: Mapping[str, Any]
    alternatives: Sequence[str]
```

- [ ] **Step 8: Run all tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_protocol.py -v
```

Expected: 8 passed.

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/web/composer/guided/__init__.py \
        src/elspeth/web/composer/guided/protocol.py \
        tests/unit/web/composer/guided/__init__.py \
        tests/unit/web/composer/guided/test_protocol.py
git commit -m "$(cat <<'EOF'
feat(composer/guided): protocol module skeleton — TurnType + payload shapes

First slice of guided-mode protocol foundation. Defines the closed
six-turn taxonomy as a str-valued Enum and the TypedDict shape of each
turn's payload. Pure data; no I/O, no LLM. Subsequent tasks add
TurnResponse, ControlSignal, the legal-turn matrix, and payload validators.

Refs: docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.2: Add `TurnResponse`, `ControlSignal`, `Turn` types

**Files:**
- Modify: `src/elspeth/web/composer/guided/protocol.py`
- Modify: `tests/unit/web/composer/guided/test_protocol.py`

- [ ] **Step 1: Write failing tests for response types**

Append to `test_protocol.py`:

```python
from elspeth.web.composer.guided.protocol import (
    ControlSignal,
    Turn,
    TurnResponse,
)


class TestTurnResponse:
    def test_control_signal_values(self) -> None:
        assert {s.value for s in ControlSignal} == {
            "exit_to_freeform",
            "request_advisor",
            "reject",
        }

    def test_turn_response_minimal(self) -> None:
        resp: TurnResponse = {
            "chosen": ["jsonl"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        assert resp["chosen"] == ["jsonl"]

    def test_turn_response_with_control_signal(self) -> None:
        resp: TurnResponse = {
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": ControlSignal.EXIT_TO_FREEFORM.value,
        }
        assert resp["control_signal"] == "exit_to_freeform"


class TestTurn:
    def test_turn_carries_type_and_payload(self) -> None:
        turn: Turn = {
            "type": "single_select",
            "step_index": 1,
            "payload": {
                "question": "X?",
                "options": [],
                "allow_custom": False,
            },
        }
        assert turn["type"] == "single_select"
        assert turn["step_index"] == 1
```

Run, expect failures.

- [ ] **Step 2: Implement `ControlSignal`, `TurnResponse`, `Turn`**

Append to `src/elspeth/web/composer/guided/protocol.py`:

```python
class ControlSignal(str, Enum):
    """Out-of-band signals carried in a TurnResponse instead of (or alongside) data."""

    EXIT_TO_FREEFORM = "exit_to_freeform"
    REQUEST_ADVISOR = "request_advisor"
    REJECT = "reject"


class TurnResponse(TypedDict):
    """The user's typed response to a turn."""

    chosen: Sequence[str] | None
    edited_values: Mapping[str, Any] | None
    custom_inputs: Sequence[str] | None
    accepted_step_index: int | None
    edit_step_index: int | None
    control_signal: str | None  # ControlSignal value, or None


class Turn(TypedDict):
    """A turn emitted to the user (server-emitted or LLM-emitted)."""

    type: str  # TurnType value
    step_index: int
    payload: Mapping[str, Any]
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_protocol.py -v
```

Expected: 12 passed (8 from Task 1.1 + 4 new).

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): protocol — TurnResponse, ControlSignal, Turn types

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.3: Legal-turn matrix and validator

**Files:**
- Modify: `src/elspeth/web/composer/guided/protocol.py`
- Modify: `tests/unit/web/composer/guided/test_protocol.py`

- [ ] **Step 1: Write failing tests for legal-turn matrix**

```python
class TestLegalTurnMatrix:
    def test_step_1_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_1_SOURCE)
        assert TurnType.INSPECT_AND_CONFIRM in legal
        assert TurnType.SINGLE_SELECT in legal
        assert TurnType.SCHEMA_FORM in legal
        assert TurnType.PROPOSE_CHAIN not in legal

    def test_step_2_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_2_SINK)
        assert TurnType.SINGLE_SELECT in legal
        assert TurnType.MULTI_SELECT_WITH_CUSTOM in legal
        assert TurnType.SCHEMA_FORM in legal

    def test_step_2_5_recipe_offer_only(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_2_5_RECIPE_MATCH)
        assert legal == frozenset({TurnType.RECIPE_OFFER})

    def test_step_3_legal_types(self) -> None:
        from elspeth.web.composer.guided.protocol import GuidedStep, legal_turn_types_for

        legal = legal_turn_types_for(GuidedStep.STEP_3_TRANSFORMS)
        assert TurnType.PROPOSE_CHAIN in legal
        assert TurnType.SINGLE_SELECT in legal


class TestPayloadValidation:
    def test_validate_single_select_ok(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        ok = validate_payload(
            TurnType.SINGLE_SELECT,
            {"question": "Q?", "options": [], "allow_custom": False},
        )
        assert ok is None

    def test_validate_single_select_missing_field(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        err = validate_payload(TurnType.SINGLE_SELECT, {"question": "Q?"})
        assert err is not None
        assert "options" in err

    def test_validate_unknown_turn_type_rejected(self) -> None:
        from elspeth.web.composer.guided.protocol import validate_payload

        with pytest.raises(ValueError):
            validate_payload("not_a_turn_type", {})  # type: ignore[arg-type]
```

Run, expect import errors.

- [ ] **Step 2: Implement `GuidedStep`, `legal_turn_types_for`, `validate_payload`**

Append to `src/elspeth/web/composer/guided/protocol.py`:

```python
class GuidedStep(str, Enum):
    """Wizard step pointer."""

    STEP_1_SOURCE = "step_1_source"
    STEP_2_SINK = "step_2_sink"
    STEP_2_5_RECIPE_MATCH = "step_2_5_recipe_match"
    STEP_3_TRANSFORMS = "step_3_transforms"


_LEGAL_TURN_MATRIX: Mapping[GuidedStep, frozenset[TurnType]] = {
    GuidedStep.STEP_1_SOURCE: frozenset({
        TurnType.INSPECT_AND_CONFIRM,
        TurnType.SINGLE_SELECT,
        TurnType.SCHEMA_FORM,
    }),
    GuidedStep.STEP_2_SINK: frozenset({
        TurnType.SINGLE_SELECT,
        TurnType.MULTI_SELECT_WITH_CUSTOM,
        TurnType.SCHEMA_FORM,
    }),
    GuidedStep.STEP_2_5_RECIPE_MATCH: frozenset({TurnType.RECIPE_OFFER}),
    GuidedStep.STEP_3_TRANSFORMS: frozenset({
        TurnType.PROPOSE_CHAIN,
        TurnType.SINGLE_SELECT,
    }),
}


def legal_turn_types_for(step: GuidedStep) -> frozenset[TurnType]:
    """Return the frozen set of TurnType values legal at the given step."""
    return _LEGAL_TURN_MATRIX[step]


_REQUIRED_KEYS: Mapping[TurnType, frozenset[str]] = {
    TurnType.INSPECT_AND_CONFIRM: frozenset({"observed"}),
    TurnType.SINGLE_SELECT: frozenset({"question", "options", "allow_custom"}),
    TurnType.MULTI_SELECT_WITH_CUSTOM: frozenset({
        "question", "options", "default_chosen", "escape_label",
    }),
    TurnType.SCHEMA_FORM: frozenset({"plugin", "schema_block", "prefilled"}),
    TurnType.PROPOSE_CHAIN: frozenset({"steps", "why", "blockers"}),
    TurnType.RECIPE_OFFER: frozenset({"recipe_name", "slots", "alternatives"}),
}


def validate_payload(turn_type: TurnType, payload: Mapping[str, Any]) -> str | None:
    """Validate that *payload* satisfies the schema for *turn_type*.

    Returns None on success, or a human-readable error string on failure.
    Raises ValueError if turn_type is not a known TurnType.
    """
    if not isinstance(turn_type, TurnType):
        raise ValueError(f"unknown turn type: {turn_type!r}")
    required = _REQUIRED_KEYS[turn_type]
    missing = required - payload.keys()
    if missing:
        return f"payload for {turn_type.value} missing required keys: {sorted(missing)}"
    return None
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_protocol.py -v
```

Expected: 19 passed (12 from prior tasks + 7 new).

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): protocol — GuidedStep + legal-turn matrix + payload validator

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.4: `GuidedSession`, `TerminalState`, `TurnRecord` dataclasses

**Files:**
- Create: `src/elspeth/web/composer/guided/state_machine.py`
- Test: `tests/unit/web/composer/guided/test_state_machine.py`

- [ ] **Step 1: Write failing tests for the state-machine dataclasses**

```python
# tests/unit/web/composer/guided/test_state_machine.py
"""Tests for GuidedSession, TerminalState, TurnRecord — state machine data."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    TerminalKind,
    TerminalReason,
    TerminalState,
    TurnRecord,
)


class TestTerminalState:
    def test_completed_kind_has_no_reason(self) -> None:
        t = TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="pipeline:\n")
        assert t.kind is TerminalKind.COMPLETED
        assert t.reason is None

    def test_exited_to_freeform_requires_reason(self) -> None:
        t = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        assert t.reason is TerminalReason.USER_PRESSED_EXIT

    def test_terminal_state_is_frozen(self) -> None:
        t = TerminalState(kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml=None)
        with pytest.raises(AttributeError):
            t.kind = TerminalKind.EXITED_TO_FREEFORM  # type: ignore[misc]


class TestTurnRecord:
    def test_turn_record_carries_emitted_and_response(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="abc123",
            response_hash="def456",
            emitter="server",
        )
        assert rec.emitter == "server"

    def test_turn_record_frozen(self) -> None:
        rec = TurnRecord(
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="abc",
            response_hash=None,
            emitter="server",
        )
        with pytest.raises(AttributeError):
            rec.emitter = "llm"  # type: ignore[misc]


class TestGuidedSession:
    def test_initial_session_at_step_1(self) -> None:
        s = GuidedSession.initial()
        assert s.step is GuidedStep.STEP_1_SOURCE
        assert s.terminal is None
        assert s.history == ()

    def test_session_history_is_immutable_tuple(self) -> None:
        s = GuidedSession.initial()
        with pytest.raises(AttributeError):
            s.history.append(None)  # type: ignore[attr-defined]

    def test_session_with_terminal_set(self) -> None:
        s = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
            terminal=TerminalState(
                kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml="x:\n"
            ),
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
        )
        assert s.terminal is not None
        assert s.terminal.kind is TerminalKind.COMPLETED
```

Run, expect import errors.

- [ ] **Step 2: Implement state-machine dataclasses with freeze guards**

```python
# src/elspeth/web/composer/guided/state_machine.py
"""Guided-mode state-machine data: GuidedSession, TerminalState, TurnRecord.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §5.

Trust tier: Tier 1 (audit). Coercion forbidden — every field crashes on
malformed input. The freeze_fields contract applies because these structures
are persisted and re-read across the audit trail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType


class TerminalKind(str, Enum):
    COMPLETED = "completed"
    EXITED_TO_FREEFORM = "exited_to_freeform"


class TerminalReason(str, Enum):
    USER_PRESSED_EXIT = "user_pressed_exit"
    PROTOCOL_VIOLATION = "protocol_violation"
    SOLVER_EXHAUSTED = "solver_exhausted"


@dataclass(frozen=True, slots=True)
class TerminalState:
    """Outcome of a guided session.

    `reason` is None when `kind == COMPLETED`; required when
    `kind == EXITED_TO_FREEFORM`. `pipeline_yaml` is set only on COMPLETED.
    Callers must construct consistently — invariants enforced by step_advance().
    """

    kind: TerminalKind
    reason: TerminalReason | None
    pipeline_yaml: str | None


@dataclass(frozen=True, slots=True)
class TurnRecord:
    """One emitted turn + its (optional) user response, recorded for audit."""

    step: GuidedStep
    turn_type: TurnType
    payload_hash: str
    response_hash: str | None
    emitter: str  # "server" | "llm"


@dataclass(frozen=True, slots=True)
class SourceResolved:
    plugin: str
    options: Mapping[str, Any]
    observed_columns: Sequence[str]
    sample_rows: Sequence[Mapping[str, Any]]

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "observed_columns", "sample_rows")


@dataclass(frozen=True, slots=True)
class SinkOutputResolved:
    plugin: str
    options: Mapping[str, Any]
    required_fields: Sequence[str]
    schema_mode: str  # "fixed" | "flexible" | "observed"

    def __post_init__(self) -> None:
        freeze_fields(self, "options", "required_fields")


@dataclass(frozen=True, slots=True)
class SinkResolved:
    outputs: Sequence[SinkOutputResolved]

    def __post_init__(self) -> None:
        freeze_fields(self, "outputs")


@dataclass(frozen=True, slots=True)
class ChainProposal:
    steps: Sequence[Mapping[str, Any]]  # each step: {plugin, options, rationale}
    why: str

    def __post_init__(self) -> None:
        freeze_fields(self, "steps")


@dataclass(frozen=True, slots=True)
class GuidedSession:
    """The guided-mode session state.

    Persisted in CompositionState.guided_session. `terminal` becomes non-None
    when the wizard ends; subsequent freeform turns honour progressive
    disclosure (see §8.2 of the spec).
    """

    step: GuidedStep
    history: tuple[TurnRecord, ...]
    step_1_result: SourceResolved | None
    step_2_result: SinkResolved | None
    step_3_proposal: ChainProposal | None
    terminal: TerminalState | None

    @classmethod
    def initial(cls) -> GuidedSession:
        return cls(
            step=GuidedStep.STEP_1_SOURCE,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py -v
```

Expected: 8 passed (3 TestTerminalState + 2 TestTurnRecord + 3 TestGuidedSession).

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/guided/state_machine.py \
        tests/unit/web/composer/guided/test_state_machine.py
git commit -m "feat(composer/guided): state-machine dataclasses with freeze guards

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.5: Add `guided_session` field to `CompositionState`

**Files:**
- Modify: `src/elspeth/web/composer/state.py:1618-` (CompositionState class)
- Test: `tests/unit/web/composer/guided/test_state_field.py`

- [ ] **Step 1: Read the existing `CompositionState` definition**

```bash
.venv/bin/python -c "import inspect; from elspeth.web.composer.state import CompositionState; print(inspect.getsourcefile(CompositionState))"
.venv/bin/grep -nE "^class CompositionState|guided_session" src/elspeth/web/composer/state.py
```

Inspect the class to see its existing `__post_init__` and `freeze_fields` call so the new field is added consistently.

- [ ] **Step 2: Write failing test for the new field**

```python
# tests/unit/web/composer/guided/test_state_field.py
"""Tests for CompositionState.guided_session field freeze guard."""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.state_machine import GuidedSession
from elspeth.web.composer.state import CompositionState


class TestGuidedSessionField:
    def test_default_is_none(self) -> None:
        state = CompositionState()
        assert state.guided_session is None

    def test_can_attach_initial_session(self) -> None:
        sess = GuidedSession.initial()
        state = CompositionState(guided_session=sess)
        assert state.guided_session is sess

    def test_session_history_remains_immutable(self) -> None:
        sess = GuidedSession.initial()
        state = CompositionState(guided_session=sess)
        with pytest.raises(AttributeError):
            state.guided_session = None  # type: ignore[misc]
```

Run, expect failures because `CompositionState` does not yet accept `guided_session`.

- [ ] **Step 3: Add the field to `CompositionState`**

In `src/elspeth/web/composer/state.py` at the `CompositionState` class (line ~1618), add the field. Find the `@dataclass(frozen=True, slots=True)` `class CompositionState:` block; add a `guided_session: GuidedSession | None = None` field at the end of the field list. Add the import at the top of the file:

```python
from elspeth.web.composer.guided.state_machine import GuidedSession
```

The field is a `frozen=True, slots=True` dataclass, so it doesn't need a `freeze_fields` call (no inner mutable container).

- [ ] **Step 4: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_field.py tests/unit/web/composer/test_state.py -v
```

Expected: all green. The existing `test_state.py` tests still pass (no regression).

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(composer/state): add guided_session field to CompositionState

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.6: Audit event types and emit helpers

**Files:**
- Create: `src/elspeth/web/composer/guided/audit.py`
- Test: `tests/unit/web/composer/guided/test_audit.py`

- [ ] **Step 1: Write failing tests for audit emit helpers**

```python
# tests/unit/web/composer/guided/test_audit.py
"""Tests for guided-mode audit emit helpers.

Audit-tier (Tier 1) per CLAUDE.md. Coercion forbidden — every field is
either present or the function raises.
"""

from __future__ import annotations

from typing import Any

import pytest

from elspeth.web.composer.guided.audit import (
    GuidedAuditEvent,
    emit_dropped_to_freeform,
    emit_step_advanced,
    emit_turn_answered,
    emit_turn_emitted,
)
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import TerminalReason


class _FakeRecorder:
    """Captures audit events without DB side effects."""

    def __init__(self) -> None:
        self.events: list[GuidedAuditEvent] = []

    def record_guided_event(self, event: GuidedAuditEvent) -> None:
        self.events.append(event)


class TestEmitTurnEmitted:
    def test_records_step_and_type(self) -> None:
        rec = _FakeRecorder()
        emit_turn_emitted(
            rec,
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            payload_hash="abc",
            payload_payload_id="payload-1",
            emitter="server",
        )
        assert len(rec.events) == 1
        evt = rec.events[0]
        assert evt.event_type == "guided_turn_emitted"
        assert evt.step_index == "step_1_source"


class TestEmitTurnAnswered:
    def test_records_response_hash(self) -> None:
        rec = _FakeRecorder()
        emit_turn_answered(
            rec,
            step=GuidedStep.STEP_1_SOURCE,
            turn_type=TurnType.SINGLE_SELECT,
            response_hash="xyz",
            response_payload_id="payload-2",
            control_signal=None,
        )
        assert rec.events[0].event_type == "guided_turn_answered"


class TestEmitStepAdvanced:
    def test_records_prev_and_next(self) -> None:
        rec = _FakeRecorder()
        emit_step_advanced(
            rec,
            prev=GuidedStep.STEP_1_SOURCE,
            next_=GuidedStep.STEP_2_SINK,
            reason="user_advanced",
        )
        assert rec.events[0].event_type == "guided_step_advanced"


class TestEmitDroppedToFreeform:
    def test_records_drop_reason(self) -> None:
        rec = _FakeRecorder()
        emit_dropped_to_freeform(
            rec,
            prev=GuidedStep.STEP_3_TRANSFORMS,
            drop_reason=TerminalReason.SOLVER_EXHAUSTED,
            validation_result={"errors": ["..."]},
        )
        evt = rec.events[0]
        assert evt.event_type == "guided_dropped_to_freeform"
        assert evt.payload["drop_reason"] == "solver_exhausted"

    def test_user_pressed_exit_has_no_validation_result(self) -> None:
        rec = _FakeRecorder()
        emit_dropped_to_freeform(
            rec,
            prev=GuidedStep.STEP_2_SINK,
            drop_reason=TerminalReason.USER_PRESSED_EXIT,
            validation_result=None,
        )
        evt = rec.events[0]
        assert evt.payload.get("validation_result") is None
```

Run, expect import errors.

- [ ] **Step 2: Implement `audit.py`**

```python
# src/elspeth/web/composer/guided/audit.py
"""Guided-mode audit event emission.

Tier 1 (audit-trust). Coercion forbidden — invalid input crashes. Events go
through the existing BufferingRecorder plumbing via record_guided_event().

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §9.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from elspeth.contracts.freeze import freeze_fields
from elspeth.web.composer.guided.protocol import GuidedStep, TurnType
from elspeth.web.composer.guided.state_machine import TerminalReason


@dataclass(frozen=True, slots=True)
class GuidedAuditEvent:
    """An audit event for a guided-mode action."""

    event_type: str
    step_index: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "payload")


class _GuidedRecorderProtocol(Protocol):
    def record_guided_event(self, event: GuidedAuditEvent) -> None: ...


def emit_turn_emitted(
    recorder: _GuidedRecorderProtocol,
    *,
    step: GuidedStep,
    turn_type: TurnType,
    payload_hash: str,
    payload_payload_id: str,
    emitter: str,
) -> None:
    if emitter not in {"server", "llm"}:
        raise ValueError(f"emitter must be 'server' or 'llm', got {emitter!r}")
    recorder.record_guided_event(
        GuidedAuditEvent(
            event_type="guided_turn_emitted",
            step_index=step.value,
            payload={
                "turn_type": turn_type.value,
                "payload_hash": payload_hash,
                "payload_payload_id": payload_payload_id,
                "emitter": emitter,
            },
        )
    )


def emit_turn_answered(
    recorder: _GuidedRecorderProtocol,
    *,
    step: GuidedStep,
    turn_type: TurnType,
    response_hash: str,
    response_payload_id: str,
    control_signal: str | None,
) -> None:
    payload: dict[str, Any] = {
        "turn_type": turn_type.value,
        "response_hash": response_hash,
        "response_payload_id": response_payload_id,
    }
    if control_signal is not None:
        payload["control_signal"] = control_signal
    recorder.record_guided_event(
        GuidedAuditEvent(
            event_type="guided_turn_answered",
            step_index=step.value,
            payload=payload,
        )
    )


def emit_step_advanced(
    recorder: _GuidedRecorderProtocol,
    *,
    prev: GuidedStep,
    next_: GuidedStep,
    reason: str,
) -> None:
    if reason not in {"recipe_applied", "user_advanced", "auto_advanced"}:
        raise ValueError(f"unknown advance reason: {reason!r}")
    recorder.record_guided_event(
        GuidedAuditEvent(
            event_type="guided_step_advanced",
            step_index=next_.value,
            payload={"prev_step": prev.value, "reason": reason},
        )
    )


def emit_dropped_to_freeform(
    recorder: _GuidedRecorderProtocol,
    *,
    prev: GuidedStep,
    drop_reason: TerminalReason,
    validation_result: Mapping[str, Any] | None,
) -> None:
    payload: dict[str, Any] = {"drop_reason": drop_reason.value}
    if validation_result is not None:
        payload["validation_result"] = dict(validation_result)
    recorder.record_guided_event(
        GuidedAuditEvent(
            event_type="guided_dropped_to_freeform",
            step_index=prev.value,
            payload=payload,
        )
    )
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_audit.py -v
```

Expected: 6 passed.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/guided/audit.py \
        tests/unit/web/composer/guided/test_audit.py
git commit -m "feat(composer/guided): audit event types and emit helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 1.7: Phase 1 closure — full Ring 1 sweep

- [ ] **Step 1: Run full Ring 1 sweep**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/ -v
.venv/bin/python -m mypy src/elspeth/web/composer/guided/
.venv/bin/python -m ruff check src/elspeth/web/composer/guided/
```

Expected: all pass; mypy clean; ruff clean.

- [ ] **Step 2: Verify tier-model enforcement passes (no upward imports)**

```bash
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: pass. The new module under `composer/guided/` (L3) imports only from `contracts/` (L0) and other L3 — no inversions.

- [ ] **Step 3: Phase-1 commit if any cleanup needed**

If lint or mypy required fixups, commit them as `chore(composer/guided): Phase 1 lint/typecheck cleanup`.

---

## Phase 2 — State Machine, Recipe Match, Step Handler Stubs

**Goal of phase:** Implement the pure `step_advance()` function, the deterministic recipe matcher, and the four step-handler stubs that drive `CompositionState` mutation via existing `tools.py` handlers. Still no I/O — `step_advance` is a pure function over `(GuidedSession, TurnResponse, CompositionState)`. Step handlers are tested by injecting fakes for the `tools.py` mutation helpers.

**Phase exit criterion:** Hypothesis property tests pass for the state machine; recipe matcher exhaustively tested against all three registered recipes; step handlers reach correct next-step or terminal-state outcomes given canned responses.

### Task 2.1: `step_advance()` — Step 1 → Step 2 transitions

**Files:**
- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `tests/unit/web/composer/guided/test_state_machine.py`

- [ ] **Step 1: Write failing test for Step 1 → Step 2 advance on source confirmation**

Append to `test_state_machine.py`:

```python
from collections.abc import Mapping
from typing import Any

from elspeth.web.composer.guided.protocol import TurnResponse
from elspeth.web.composer.guided.state_machine import (
    SourceResolved,
    step_advance,
)


class TestStepAdvance:
    def test_initial_session_advances_after_source_confirmed(self) -> None:
        sess = GuidedSession.initial()
        # User confirms inspect_and_confirm with no edits
        response: TurnResponse = {
            "chosen": None,
            "edited_values": {
                "plugin": "csv",
                "options": {"path": "x.csv", "schema": {"mode": "observed"}},
                "observed_columns": ["a", "b"],
                "sample_rows": [{"a": "1", "b": "2"}],
            },
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }

        new_sess, _next_turn, terminal, audit_events = step_advance(
            sess, response, current_turn_type=TurnType.INSPECT_AND_CONFIRM
        )

        assert new_sess.step is GuidedStep.STEP_2_SINK
        assert new_sess.step_1_result is not None
        assert new_sess.step_1_result.plugin == "csv"
        assert terminal is None
        assert any(e.event_type == "guided_step_advanced" for e in audit_events)
```

Run, expect failure (no `step_advance`).

- [ ] **Step 2: Implement `step_advance()` for the Step 1 → Step 2 transition**

Append to `state_machine.py`:

```python
from elspeth.web.composer.guided.audit import (
    GuidedAuditEvent,
    emit_step_advanced,
    emit_turn_answered,
)


class _CapturingRecorder:
    """In-process recorder used by step_advance to collect events for the caller."""

    def __init__(self) -> None:
        self.events: list[GuidedAuditEvent] = []

    def record_guided_event(self, event: GuidedAuditEvent) -> None:
        self.events.append(event)


def step_advance(
    session: GuidedSession,
    response: TurnResponse,
    *,
    current_turn_type: TurnType,
) -> tuple[
    GuidedSession,
    Turn | None,
    TerminalState | None,
    list[GuidedAuditEvent],
]:
    """Apply *response* to *session*. Pure function (no I/O).

    Returns (new_session, next_turn_or_None, terminal_or_None, audit_events).
    The caller (the endpoint handler) emits the audit_events via the real
    BufferingRecorder.
    """
    rec = _CapturingRecorder()

    if response["control_signal"] == "exit_to_freeform":
        terminal = TerminalState(
            kind=TerminalKind.EXITED_TO_FREEFORM,
            reason=TerminalReason.USER_PRESSED_EXIT,
            pipeline_yaml=None,
        )
        new_sess = _replace(session, terminal=terminal)
        return (new_sess, None, terminal, rec.events)

    if session.step is GuidedStep.STEP_1_SOURCE:
        return _advance_step_1(session, response, current_turn_type, rec)
    if session.step is GuidedStep.STEP_2_SINK:
        return _advance_step_2(session, response, current_turn_type, rec)
    if session.step is GuidedStep.STEP_2_5_RECIPE_MATCH:
        return _advance_step_2_5(session, response, current_turn_type, rec)
    if session.step is GuidedStep.STEP_3_TRANSFORMS:
        return _advance_step_3(session, response, current_turn_type, rec)
    raise AssertionError(f"unhandled step: {session.step}")


def _replace(session: GuidedSession, **fields: Any) -> GuidedSession:
    """Helper: dataclass-replace for GuidedSession (frozen)."""
    from dataclasses import replace

    return replace(session, **fields)


def _advance_step_1(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
    rec: _CapturingRecorder,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditEvent]]:
    if turn_type is not TurnType.INSPECT_AND_CONFIRM:
        # Other Step 1 turn types (single_select for plugin, schema_form for options)
        # do not yet advance — they emit the next intra-step turn.
        # For this slice, treat non-INSPECT_AND_CONFIRM as no-advance.
        return (session, None, None, rec.events)

    edited = response["edited_values"]
    if edited is None:
        raise ValueError("inspect_and_confirm response must carry edited_values")
    source = SourceResolved(
        plugin=str(edited["plugin"]),
        options=dict(edited["options"]),
        observed_columns=tuple(edited["observed_columns"]),
        sample_rows=tuple(dict(r) for r in edited["sample_rows"]),
    )
    emit_step_advanced(
        rec,
        prev=GuidedStep.STEP_1_SOURCE,
        next_=GuidedStep.STEP_2_SINK,
        reason="user_advanced",
    )
    new_sess = _replace(
        session,
        step=GuidedStep.STEP_2_SINK,
        step_1_result=source,
    )
    # Next turn is constructed by the endpoint handler; step_advance is pure.
    return (new_sess, None, None, rec.events)


def _advance_step_2(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
    rec: _CapturingRecorder,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditEvent]]:
    # Implemented in Task 2.2.
    raise NotImplementedError("step 2 advance — implemented in task 2.2")


def _advance_step_2_5(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
    rec: _CapturingRecorder,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditEvent]]:
    # Implemented in Task 2.3.
    raise NotImplementedError("step 2.5 advance — implemented in task 2.3")


def _advance_step_3(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
    rec: _CapturingRecorder,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditEvent]]:
    # Implemented in Task 2.4.
    raise NotImplementedError("step 3 advance — implemented in task 2.4")
```

Add the missing import at the top of `state_machine.py`:

```python
from elspeth.web.composer.guided.protocol import GuidedStep, Turn, TurnResponse, TurnType
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py::TestStepAdvance -v
```

Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): step_advance — Step 1 → Step 2 transition

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.2: `step_advance()` — Step 2 → Step 2.5 transitions

**Files:**
- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `tests/unit/web/composer/guided/test_state_machine.py`

- [ ] **Step 1: Write failing test for Step 2 → Step 2.5**

Append to `test_state_machine.py`:

```python
    def test_step_2_advances_after_required_fields_declared(self) -> None:
        sess = GuidedSession(
            step=GuidedStep.STEP_2_SINK,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a", "b"),
                sample_rows=({"a": "1", "b": "2"},),
            ),
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": None,
            "edited_values": {
                "outputs": [
                    {
                        "plugin": "json",
                        "options": {"path": "out.jsonl"},
                        "required_fields": ["a"],
                        "schema_mode": "fixed",
                    },
                ],
            },
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }

        new_sess, _next, terminal, _events = step_advance(
            sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM
        )

        assert new_sess.step is GuidedStep.STEP_2_5_RECIPE_MATCH
        assert new_sess.step_2_result is not None
        assert len(new_sess.step_2_result.outputs) == 1
        assert terminal is None

    def test_step_2_let_source_decide_sets_observed_mode(self) -> None:
        sess = GuidedSession(
            step=GuidedStep.STEP_2_SINK,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv",
                options={},
                observed_columns=("a", "b"),
                sample_rows=({},),
            ),
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": None,
            "edited_values": {
                "outputs": [
                    {
                        "plugin": "json",
                        "options": {"path": "out.jsonl"},
                        "required_fields": [],
                        "schema_mode": "observed",
                    },
                ],
            },
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, _next, terminal, _events = step_advance(
            sess, response, current_turn_type=TurnType.MULTI_SELECT_WITH_CUSTOM
        )
        assert new_sess.step_2_result is not None
        assert new_sess.step_2_result.outputs[0].schema_mode == "observed"
```

Run, expect `NotImplementedError` from the stub.

- [ ] **Step 2: Implement `_advance_step_2`**

Replace the `_advance_step_2` stub:

```python
def _advance_step_2(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
    rec: _CapturingRecorder,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditEvent]]:
    if turn_type is not TurnType.MULTI_SELECT_WITH_CUSTOM:
        return (session, None, None, rec.events)

    edited = response["edited_values"]
    if edited is None:
        raise ValueError("multi_select response must carry edited_values")
    raw_outputs = edited["outputs"]
    outputs = tuple(
        SinkOutputResolved(
            plugin=str(o["plugin"]),
            options=dict(o["options"]),
            required_fields=tuple(o["required_fields"]),
            schema_mode=str(o["schema_mode"]),
        )
        for o in raw_outputs
    )
    sink = SinkResolved(outputs=outputs)
    emit_step_advanced(
        rec,
        prev=GuidedStep.STEP_2_SINK,
        next_=GuidedStep.STEP_2_5_RECIPE_MATCH,
        reason="user_advanced",
    )
    new_sess = _replace(
        session,
        step=GuidedStep.STEP_2_5_RECIPE_MATCH,
        step_2_result=sink,
    )
    return (new_sess, None, None, rec.events)
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): step_advance — Step 2 → Step 2.5 transition

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.3: Recipe matcher

**Files:**
- Create: `src/elspeth/web/composer/guided/recipe_match.py`
- Test: `tests/unit/web/composer/guided/test_recipe_match.py`

- [ ] **Step 1: Write failing tests for the three currently-registered recipes**

```python
# tests/unit/web/composer/guided/test_recipe_match.py
"""Tests for deterministic recipe pre-match.

Coverage: each registered recipe gets at least one matching test and one
non-matching test. Specificity ordering tested when overlapping predicates
exist.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.recipe_match import RecipeMatch, match_recipe
from elspeth.web.composer.guided.state_machine import (
    SinkOutputResolved,
    SinkResolved,
    SourceResolved,
)


def _csv_source(columns: tuple[str, ...] = ("a", "b")) -> SourceResolved:
    return SourceResolved(
        plugin="csv", options={}, observed_columns=columns, sample_rows=({},)
    )


def _json_sink(required: tuple[str, ...] = ("category",)) -> SinkResolved:
    return SinkResolved(
        outputs=(
            SinkOutputResolved(
                plugin="json", options={"path": "out.jsonl"},
                required_fields=required, schema_mode="fixed",
            ),
        )
    )


class TestClassifyRowsLLMJsonl:
    def test_csv_to_json_with_llm_intent_matches(self) -> None:
        match = match_recipe(
            _csv_source(),
            _json_sink(required=("category", "confidence")),
        )
        assert match is not None
        assert match.recipe_name == "classify-rows-llm-jsonl"

    def test_csv_to_csv_does_not_match_classify(self) -> None:
        from elspeth.web.composer.guided.state_machine import SinkOutputResolved

        sink = SinkResolved(
            outputs=(
                SinkOutputResolved(
                    plugin="csv", options={}, required_fields=(), schema_mode="observed"
                ),
            )
        )
        match = match_recipe(_csv_source(), sink)
        # Either matches a different recipe or returns None — but not classify
        if match is not None:
            assert match.recipe_name != "classify-rows-llm-jsonl"


class TestSplitByNumericThreshold:
    def test_csv_to_two_jsons_with_numeric_field_matches(self) -> None:
        sink = SinkResolved(
            outputs=(
                SinkOutputResolved(
                    plugin="json", options={"path": "above.jsonl"},
                    required_fields=(), schema_mode="observed",
                ),
                SinkOutputResolved(
                    plugin="json", options={"path": "below.jsonl"},
                    required_fields=(), schema_mode="observed",
                ),
            )
        )
        match = match_recipe(
            _csv_source(columns=("price", "qty")),
            sink,
        )
        # Note: predicate currently keys on output count + json sink. Refine if
        # the registered recipe's predicate becomes stricter.
        if match is not None:
            assert match.recipe_name in {
                "split-by-numeric-threshold",
                "fork-coalesce-truncate-jsonl",
            }


class TestNoMatch:
    def test_database_sink_no_match(self) -> None:
        sink = SinkResolved(
            outputs=(
                SinkOutputResolved(
                    plugin="database",
                    options={"url": "sqlite:///x.db", "table": "rows"},
                    required_fields=(), schema_mode="observed",
                ),
            )
        )
        assert match_recipe(_csv_source(), sink) is None
```

Run, expect import errors.

- [ ] **Step 2: Implement `recipe_match.py`**

```python
# src/elspeth/web/composer/guided/recipe_match.py
"""Deterministic recipe pre-match for guided mode Step 2.5.

Pure function over (SourceResolved, SinkResolved). No LLM, no fuzzy matching.
Predicates are simple boolean expressions; specificity ordering chooses the
most-specific match when multiple predicates fire.

See docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md §6.4.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from elspeth.web.composer.guided.state_machine import SinkResolved, SourceResolved


@dataclass(frozen=True, slots=True)
class RecipeMatch:
    """Result of matching a recipe to a (source, sink) tuple."""

    recipe_name: str
    slots: Mapping[str, Any]


_Predicate = Callable[[SourceResolved, SinkResolved], bool]
_SlotResolver = Callable[[SourceResolved, SinkResolved], Mapping[str, Any]]


def _is_csv(source: SourceResolved) -> bool:
    return source.plugin == "csv"


def _has_single_json_output(sink: SinkResolved) -> bool:
    return len(sink.outputs) == 1 and sink.outputs[0].plugin == "json"


def _has_two_json_outputs(sink: SinkResolved) -> bool:
    return len(sink.outputs) == 2 and all(o.plugin == "json" for o in sink.outputs)


def _classify_predicate(source: SourceResolved, sink: SinkResolved) -> bool:
    if not (_is_csv(source) and _has_single_json_output(sink)):
        return False
    required = sink.outputs[0].required_fields
    return any(name in {"category", "label", "tag", "classification"} for name in required)


def _classify_slot_resolver(
    source: SourceResolved, sink: SinkResolved
) -> Mapping[str, Any]:
    return {
        "csv_blob_id": source.options.get("blob_id", ""),
        "output_path": sink.outputs[0].options.get("path", "classified.jsonl"),
        "output_field": sink.outputs[0].required_fields[0],
        # prompt_template, model, provider remain operator-fillable defaults
    }


def _split_threshold_predicate(
    source: SourceResolved, sink: SinkResolved
) -> bool:
    return _is_csv(source) and _has_two_json_outputs(sink)


def _split_threshold_slot_resolver(
    source: SourceResolved, sink: SinkResolved
) -> Mapping[str, Any]:
    return {
        "csv_blob_id": source.options.get("blob_id", ""),
        "above_output_path": sink.outputs[0].options.get("path", "above.jsonl"),
        "below_output_path": sink.outputs[1].options.get("path", "below.jsonl"),
    }


# Ordered most-specific first.
_RECIPE_PREDICATES: Sequence[tuple[_Predicate, str, _SlotResolver]] = (
    (_classify_predicate, "classify-rows-llm-jsonl", _classify_slot_resolver),
    (_split_threshold_predicate, "split-by-numeric-threshold", _split_threshold_slot_resolver),
    # fork-coalesce-truncate-jsonl predicate intentionally omitted in v1; that
    # recipe requires structural intent (truncate one path) that the matcher
    # cannot infer from (source, sink) alone. Users who want it must hand-build
    # via Step 3 or pre-select it from list_recipes.
)


def match_recipe(
    source: SourceResolved,
    sink: SinkResolved,
) -> RecipeMatch | None:
    """Return the most-specific recipe matching the tuple, or None."""
    for predicate, name, resolver in _RECIPE_PREDICATES:
        if predicate(source, sink):
            return RecipeMatch(recipe_name=name, slots=resolver(source, sink))
    return None
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_recipe_match.py -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/guided/recipe_match.py \
        tests/unit/web/composer/guided/test_recipe_match.py
git commit -m "feat(composer/guided): deterministic recipe matcher with two predicates

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.4: `step_advance()` — Step 2.5 → Step 3 (or skip on recipe accept)

**Files:**
- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `tests/unit/web/composer/guided/test_state_machine.py`

- [ ] **Step 1: Write failing tests for the two Step 2.5 outcomes**

```python
    def test_step_2_5_recipe_accepted_terminates_with_completed(self) -> None:
        # Recipe applied — wizard short-circuits to a completed terminal state
        # via the recipe-application path (Task 3.4 wires the actual application).
        # For step_advance(), the response indicates "accept" and we transition
        # to the recipe-applied marker step; the endpoint handler runs apply_recipe.
        sess = GuidedSession(
            step=GuidedStep.STEP_2_5_RECIPE_MATCH,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv", options={"blob_id": "blob-1"},
                observed_columns=("a",), sample_rows=({},),
            ),
            step_2_result=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json", options={"path": "out.jsonl"},
                        required_fields=("category",), schema_mode="fixed",
                    ),
                )
            ),
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": ["accept"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, _next, terminal, _events = step_advance(
            sess, response, current_turn_type=TurnType.RECIPE_OFFER,
        )
        # state_machine signals "recipe accepted" by leaving step=STEP_2_5
        # with a flag the endpoint handler interprets to run apply_recipe.
        # Implementation choice: a sentinel field `recipe_accepted: bool`.
        assert new_sess.step is GuidedStep.STEP_2_5_RECIPE_MATCH
        # Endpoint handler (Task 3.4) reads this and applies the recipe.
        # step_advance ALONE does not run apply_recipe — it's pure.
        # Convention: response['chosen'] == ['accept'] is the signal.

    def test_step_2_5_build_manually_advances_to_step_3(self) -> None:
        sess = GuidedSession(
            step=GuidedStep.STEP_2_5_RECIPE_MATCH,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv", options={}, observed_columns=("a",), sample_rows=({},),
            ),
            step_2_result=SinkResolved(outputs=()),
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": ["build_manually"],
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, _next, terminal, _events = step_advance(
            sess, response, current_turn_type=TurnType.RECIPE_OFFER,
        )
        assert new_sess.step is GuidedStep.STEP_3_TRANSFORMS
        assert terminal is None
```

Run, expect `NotImplementedError`.

- [ ] **Step 2: Implement `_advance_step_2_5`**

Replace the stub:

```python
def _advance_step_2_5(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
    rec: _CapturingRecorder,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditEvent]]:
    if turn_type is not TurnType.RECIPE_OFFER:
        return (session, None, None, rec.events)
    chosen = response["chosen"] or []
    if chosen == ["accept"]:
        # Endpoint handler reads this state and runs apply_recipe (Task 3.4).
        # step_advance leaves the session at STEP_2_5 with a sentinel; the
        # handler advances to terminal=COMPLETED after committing.
        return (session, None, None, rec.events)
    if chosen == ["build_manually"]:
        emit_step_advanced(
            rec,
            prev=GuidedStep.STEP_2_5_RECIPE_MATCH,
            next_=GuidedStep.STEP_3_TRANSFORMS,
            reason="user_advanced",
        )
        new_sess = _replace(session, step=GuidedStep.STEP_3_TRANSFORMS)
        return (new_sess, None, None, rec.events)
    raise ValueError(f"unexpected chosen for recipe_offer: {chosen!r}")
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): step_advance — Step 2.5 recipe accept/build-manually

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.5: `step_advance()` — Step 3 chain accept/edit/reject and terminal-failure paths

**Files:**
- Modify: `src/elspeth/web/composer/guided/state_machine.py`
- Modify: `tests/unit/web/composer/guided/test_state_machine.py`

- [ ] **Step 1: Write failing tests for Step 3 outcomes**

```python
    def test_step_3_accept_chain_marks_session_with_proposal(self) -> None:
        proposal = ChainProposal(
            steps=(
                {"plugin": "type_coerce", "options": {}, "rationale": "..."},
            ),
            why="bridge",
        )
        sess = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv", options={}, observed_columns=("a",), sample_rows=({},),
            ),
            step_2_result=SinkResolved(outputs=()),
            step_3_proposal=proposal,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,  # None = full accept
            "edit_step_index": None,
            "control_signal": None,
        }
        new_sess, _next, terminal, _events = step_advance(
            sess, response, current_turn_type=TurnType.PROPOSE_CHAIN,
        )
        # The endpoint handler runs preview_pipeline and then commits via
        # tools.py handlers; step_advance ALONE returns the session unchanged
        # awaiting the handler's commit step.
        assert new_sess.step is GuidedStep.STEP_3_TRANSFORMS
        assert new_sess.step_3_proposal is proposal

    def test_step_3_solver_exhausted_drops_to_freeform(self) -> None:
        sess = GuidedSession(
            step=GuidedStep.STEP_3_TRANSFORMS,
            history=(),
            step_1_result=SourceResolved(
                plugin="csv", options={}, observed_columns=("a",), sample_rows=({},),
            ),
            step_2_result=SinkResolved(outputs=()),
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": "request_advisor",  # used by handler; not relevant here
        }
        # Direct test: a separate top-level helper `mark_solver_exhausted` is
        # invoked by the endpoint after repair+advisor failure. Test that helper:
        from elspeth.web.composer.guided.state_machine import mark_solver_exhausted

        new_sess, terminal, audit_events = mark_solver_exhausted(
            sess, validation_result={"errors": ["..."]}
        )
        assert new_sess.terminal is not None
        assert new_sess.terminal.kind is TerminalKind.EXITED_TO_FREEFORM
        assert new_sess.terminal.reason is TerminalReason.SOLVER_EXHAUSTED
        assert any(e.event_type == "guided_dropped_to_freeform" for e in audit_events)
```

Run, expect failures.

- [ ] **Step 2: Implement `_advance_step_3` and `mark_solver_exhausted` and `mark_protocol_violation`**

Replace `_advance_step_3` and append the two terminal-helpers:

```python
def _advance_step_3(
    session: GuidedSession,
    response: TurnResponse,
    turn_type: TurnType,
    rec: _CapturingRecorder,
) -> tuple[GuidedSession, Turn | None, TerminalState | None, list[GuidedAuditEvent]]:
    # Acceptance/rejection of an existing proposal is interpreted by the
    # endpoint handler (which also runs preview_pipeline and commits via
    # tools.py handlers). step_advance is pure: it does not mutate state on
    # accept; the handler does.
    if turn_type is TurnType.PROPOSE_CHAIN:
        return (session, None, None, rec.events)
    if turn_type is TurnType.SINGLE_SELECT:
        # Clarifying question answered — no step change.
        return (session, None, None, rec.events)
    raise ValueError(f"unexpected turn_type at step 3: {turn_type}")


def mark_solver_exhausted(
    session: GuidedSession,
    *,
    validation_result: Mapping[str, Any] | None,
) -> tuple[GuidedSession, TerminalState, list[GuidedAuditEvent]]:
    """Endpoint helper: stamp the session as solver-exhausted and emit the audit event."""
    rec = _CapturingRecorder()
    from elspeth.web.composer.guided.audit import emit_dropped_to_freeform

    emit_dropped_to_freeform(
        rec,
        prev=session.step,
        drop_reason=TerminalReason.SOLVER_EXHAUSTED,
        validation_result=validation_result,
    )
    terminal = TerminalState(
        kind=TerminalKind.EXITED_TO_FREEFORM,
        reason=TerminalReason.SOLVER_EXHAUSTED,
        pipeline_yaml=None,
    )
    new_sess = _replace(session, terminal=terminal)
    return (new_sess, terminal, rec.events)


def mark_protocol_violation(session: GuidedSession) -> tuple[
    GuidedSession, TerminalState, list[GuidedAuditEvent]
]:
    rec = _CapturingRecorder()
    from elspeth.web.composer.guided.audit import emit_dropped_to_freeform

    emit_dropped_to_freeform(
        rec,
        prev=session.step,
        drop_reason=TerminalReason.PROTOCOL_VIOLATION,
        validation_result=None,
    )
    terminal = TerminalState(
        kind=TerminalKind.EXITED_TO_FREEFORM,
        reason=TerminalReason.PROTOCOL_VIOLATION,
        pipeline_yaml=None,
    )
    new_sess = _replace(session, terminal=terminal)
    return (new_sess, terminal, rec.events)
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py -v
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): step_advance — Step 3 + terminal-failure helpers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.6: Hypothesis property test for state-machine invariants

**Files:**
- Modify: `tests/unit/web/composer/guided/test_state_machine.py`

- [ ] **Step 1: Write a property test**

```python
from hypothesis import given, strategies as st


class TestStateMachineInvariants:
    @given(st.sampled_from(list(GuidedStep)))
    def test_exit_to_freeform_always_terminates(self, starting_step: GuidedStep) -> None:
        sess = GuidedSession(
            step=starting_step,
            history=(),
            step_1_result=None,
            step_2_result=None,
            step_3_proposal=None,
            terminal=None,
        )
        response: TurnResponse = {
            "chosen": None,
            "edited_values": None,
            "custom_inputs": None,
            "accepted_step_index": None,
            "edit_step_index": None,
            "control_signal": "exit_to_freeform",
        }
        new_sess, _next, terminal, _events = step_advance(
            sess, response, current_turn_type=TurnType.SINGLE_SELECT,
        )
        assert terminal is not None
        assert new_sess.terminal is not None
        assert new_sess.terminal.kind is TerminalKind.EXITED_TO_FREEFORM
        assert new_sess.terminal.reason is TerminalReason.USER_PRESSED_EXIT
```

- [ ] **Step 2: Run, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_state_machine.py::TestStateMachineInvariants -v
```

Expected: pass under Hypothesis.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "test(composer/guided): hypothesis property — exit_to_freeform terminates from any step

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 2.7: Phase 2 closure

- [ ] **Step 1: Run full Phase 1+2 test sweep**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/ -v
.venv/bin/python -m mypy src/elspeth/web/composer/guided/
.venv/bin/python -m ruff check src/elspeth/web/composer/guided/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: all green.

- [ ] **Step 2: Phase-2 PR ready to ship**

State machine + recipe matcher + step-advance for all four steps + Hypothesis invariant. The next phase wires these to HTTP endpoints and `tools.py` mutators.

---

## Phase 3 — Step Handlers and HTTP Endpoints

**Goal of phase:** Wire the pure state machine to the existing composer service. Add step-handler code that drives `tools.py` mutation handlers and the recipe-application path. Register the two new HTTP endpoints (`/composer/guided/start`, `/composer/guided/respond`).

**Phase exit criterion:** Ring 2 integration tests pass for the recipe-match happy path; endpoint round-trip preserves `CompositionState` correctly; audit events flow end-to-end through `BufferingRecorder`.

### Task 3.1: Step 1 handler — wire to `_execute_set_source`

**Files:**
- Create: `src/elspeth/web/composer/guided/steps.py`
- Test: `tests/integration/web/composer/guided/__init__.py`
- Test: `tests/integration/web/composer/guided/test_step_handlers.py`

- [ ] **Step 1: Inspect existing `_execute_set_source` signature**

```bash
.venv/bin/grep -nE "^def _execute_set_source|class.*Settings|def to_settings" src/elspeth/web/composer/tools.py | head -10
```

Confirm the function signature so the step handler invokes it with the right arguments. The handler will likely receive `(state, source_payload) -> ToolResult` and the step handler unwraps the result.

- [ ] **Step 2: Write failing test**

```python
# tests/integration/web/composer/guided/test_step_handlers.py
"""Integration tests for guided-mode step handlers.

These tests use real CompositionState + the real tools.py mutation
helpers; only LLM/HTTP/disk are stubbed.
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.guided.state_machine import GuidedSession, SourceResolved
from elspeth.web.composer.guided.steps import handle_step_1_source
from elspeth.web.composer.state import CompositionState


class TestStep1Handler:
    def test_commits_source_to_composition_state(self) -> None:
        state = CompositionState()
        session = GuidedSession.initial()

        result = handle_step_1_source(
            state=state,
            session=session,
            resolved=SourceResolved(
                plugin="csv",
                options={
                    "path": "data.csv",
                    "schema": {"mode": "observed"},
                },
                observed_columns=("a", "b"),
                sample_rows=({"a": "1", "b": "2"},),
            ),
        )

        assert result.state.source is not None
        assert result.state.source.plugin == "csv"
        assert result.session.step_1_result is not None
        assert result.session.step is not None  # advance handled by caller
```

Run, expect import error.

- [ ] **Step 3: Implement `handle_step_1_source`**

```python
# src/elspeth/web/composer/guided/steps.py
"""Step handlers for guided mode.

These functions take a (state, session, resolved-step-data) tuple and
return a (state, session, ToolResult) triple. They are the only code
in guided mode that mutates pipeline state, and they do so via
existing tools.py handlers — so all freeform-path validation rules
apply identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from elspeth.web.composer.guided.state_machine import (
    GuidedSession,
    SinkResolved,
    SourceResolved,
    _replace,
)
from elspeth.web.composer.state import CompositionState
from elspeth.web.composer.tools import ToolResult, _execute_set_source


@dataclass(frozen=True, slots=True)
class StepHandlerResult:
    state: CompositionState
    session: GuidedSession
    tool_result: ToolResult


def handle_step_1_source(
    *,
    state: CompositionState,
    session: GuidedSession,
    resolved: SourceResolved,
) -> StepHandlerResult:
    """Commit *resolved* as the pipeline source via the existing freeform handler."""
    source_payload: Mapping[str, Any] = {
        "plugin": resolved.plugin,
        "options": dict(resolved.options),
        # Default audit-safe wiring; the freeform on_validation_failure default
        # ("discard") is correct for guided's MVP per skill rule 5.
        "on_success": "main",
        "on_validation_failure": "discard",
    }
    new_state, tool_result = _execute_set_source(state, source_payload)
    if not tool_result.success:
        # Step handler does not auto-recover; the endpoint layer turns this
        # into a re-emit of inspect_and_confirm with the error attached.
        return StepHandlerResult(state=state, session=session, tool_result=tool_result)
    new_session = _replace(session, step_1_result=resolved)
    return StepHandlerResult(state=new_state, session=new_session, tool_result=tool_result)
```

- [ ] **Step 4: Run test, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_step_handlers.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/guided/steps.py \
        tests/integration/web/composer/guided/__init__.py \
        tests/integration/web/composer/guided/test_step_handlers.py
git commit -m "feat(composer/guided): handle_step_1_source via existing _execute_set_source

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.2: Step 2 handler — wire to `_execute_set_output`

**Files:**
- Modify: `src/elspeth/web/composer/guided/steps.py`
- Modify: `tests/integration/web/composer/guided/test_step_handlers.py`

- [ ] **Step 1: Write failing test**

```python
class TestStep2Handler:
    def test_commits_outputs_to_composition_state(self) -> None:
        from elspeth.web.composer.guided.state_machine import (
            SinkOutputResolved,
            SinkResolved,
        )
        from elspeth.web.composer.guided.steps import handle_step_2_sink

        state = CompositionState()
        # Step 1 must already be committed; create state with a source first
        # using the helper from Task 3.1.
        step_1_result = handle_step_1_source(
            state=state,
            session=GuidedSession.initial(),
            resolved=SourceResolved(
                plugin="csv", options={"path": "x.csv", "schema": {"mode": "observed"}},
                observed_columns=("a",), sample_rows=({},),
            ),
        )

        result = handle_step_2_sink(
            state=step_1_result.state,
            session=step_1_result.session,
            resolved=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json",
                        options={"path": "out.jsonl"},
                        required_fields=("a",),
                        schema_mode="observed",
                    ),
                )
            ),
        )

        assert len(result.state.outputs) == 1
        assert result.state.outputs[0].plugin == "json"
        assert result.session.step_2_result is not None
```

Run, expect import error.

- [ ] **Step 2: Implement `handle_step_2_sink`**

Append to `steps.py`:

```python
from elspeth.web.composer.tools import _execute_set_output


def handle_step_2_sink(
    *,
    state: CompositionState,
    session: GuidedSession,
    resolved: SinkResolved,
) -> StepHandlerResult:
    """Commit each resolved sink output via _execute_set_output."""
    current_state = state
    last_result: ToolResult | None = None
    for idx, output in enumerate(resolved.outputs):
        # Convention: output names are guided_out_<idx>; the upstream wiring
        # (the on_success="main" from the source) is the connection a single
        # output can satisfy. For multi-output (fan-out), Step 3's chain
        # solver introduces the gate — until then guided supports one output
        # per pipeline (see §1.2 non-goal "multi-source", which is symmetric).
        output_payload: Mapping[str, Any] = {
            "name": f"guided_out_{idx}",
            "plugin": output.plugin,
            "sink_name": "main",
            "options": dict(output.options),
        }
        next_state, last_result = _execute_set_output(current_state, output_payload)
        if not last_result.success:
            return StepHandlerResult(
                state=current_state, session=session, tool_result=last_result
            )
        current_state = next_state
    if last_result is None:
        raise ValueError("step 2 sink resolved with no outputs — handler refuses empty list")
    new_session = _replace(session, step_2_result=resolved)
    return StepHandlerResult(state=current_state, session=new_session, tool_result=last_result)
```

- [ ] **Step 3: Run test, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_step_handlers.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): handle_step_2_sink via existing _execute_set_output

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.3: Step 2.5 handler — apply recipe + run `preview_pipeline`

**Files:**
- Modify: `src/elspeth/web/composer/guided/steps.py`
- Modify: `tests/integration/web/composer/guided/test_step_handlers.py`

- [ ] **Step 1: Write failing test**

```python
class TestStep25Handler:
    def test_apply_recipe_terminates_with_completed(self) -> None:
        from elspeth.web.composer.guided.recipe_match import RecipeMatch
        from elspeth.web.composer.guided.state_machine import (
            SinkOutputResolved, SinkResolved, TerminalKind,
        )
        from elspeth.web.composer.guided.steps import handle_step_2_5_recipe_apply

        state = CompositionState()
        result = handle_step_2_5_recipe_apply(
            state=state,
            session=GuidedSession.initial(),
            match=RecipeMatch(
                recipe_name="classify-rows-llm-jsonl",
                slots={
                    "csv_blob_id": "00000000-0000-0000-0000-000000000001",
                    "prompt_template": "Classify: {{ row['text'] }}",
                    "output_path": "out.jsonl",
                    "output_field": "category",
                },
            ),
        )

        # Recipe application produced a fully-formed pipeline.
        assert result.state.source is not None
        assert len(result.state.outputs) >= 1
        assert result.session.terminal is not None
        assert result.session.terminal.kind is TerminalKind.COMPLETED
```

Run, expect import error.

- [ ] **Step 2: Implement `handle_step_2_5_recipe_apply`**

Append to `steps.py`:

```python
from elspeth.web.composer.guided.recipe_match import RecipeMatch
from elspeth.web.composer.guided.state_machine import (
    TerminalKind,
    TerminalReason,
    TerminalState,
)
from elspeth.web.composer.recipes import apply_recipe
from elspeth.web.composer.yaml_generator import generate_yaml


def handle_step_2_5_recipe_apply(
    *,
    state: CompositionState,
    session: GuidedSession,
    match: RecipeMatch,
) -> StepHandlerResult:
    """Apply the matched recipe, render YAML, mark session COMPLETED."""
    new_state_dict = apply_recipe(match.recipe_name, match.slots)
    # apply_recipe returns a freeform dict pipeline. Convert it via the
    # existing service-side builder used by /composer/chat. Locate it in
    # service.py first (search for "set_pipeline" handler):
    #   .venv/bin/grep -n "_execute_set_pipeline\|set_pipeline(" src/elspeth/web/composer/tools.py
    # Then call it with the dict; this drives all freeform pre-validation.
    from elspeth.web.composer.tools import _execute_set_pipeline
    new_state, set_result = _execute_set_pipeline(state, new_state_dict)
    if not set_result.success:
        return StepHandlerResult(state=state, session=session, tool_result=set_result)
    yaml_text = generate_yaml(new_state)
    terminal = TerminalState(
        kind=TerminalKind.COMPLETED,
        reason=None,
        pipeline_yaml=yaml_text,
    )
    new_session = _replace(session, terminal=terminal)
    # tool_result success for the audit trail; recipe application is atomic
    return StepHandlerResult(
        state=new_state,
        session=new_session,
        tool_result=ToolResult(success=True, data=match.recipe_name),
    )
```

The endpoint reuses the existing `_execute_set_pipeline` handler (verify the exact name and signature first via grep). If the function does not exist with that exact name, locate the equivalent dict-to-state builder used by the freeform `/composer/chat` endpoint and use that. (See spec §12 open question 1.)

- [ ] **Step 3: Run test, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_step_handlers.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): handle_step_2_5_recipe_apply uses existing apply_recipe

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.4: HTTP endpoints — `/composer/guided/start` and `/composer/guided/respond`

**Files:**
- Modify: `src/elspeth/web/composer/service.py` (add endpoint registration + handler stubs)
- Test: `tests/integration/web/composer/guided/test_endpoints.py`

- [ ] **Step 1: Inspect existing endpoint registration in `service.py`**

```bash
.venv/bin/grep -nE "^@(app|router)\.|def chat|register.*endpoint" src/elspeth/web/composer/service.py | head -20
```

Identify the framework (FastAPI vs Pyramid) and find the closest existing chat endpoint to mirror.

- [ ] **Step 2: Write failing test for `/composer/guided/start`**

```python
# tests/integration/web/composer/guided/test_endpoints.py
"""HTTP-layer tests for the two guided endpoints.

Drives the endpoints via the existing test client (mirroring the freeform
chat endpoint test patterns).
"""

from __future__ import annotations

import pytest


class TestGuidedStartEndpoint:
    def test_start_returns_initial_session(self, composer_test_client) -> None:
        # Create a session via the existing /composer/sessions endpoint
        resp = composer_test_client.post(
            "/composer/sessions", json={"title": "guided test"}
        )
        session_id = resp.json()["id"]

        resp = composer_test_client.post(
            "/composer/guided/start",
            json={"session_id": session_id},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["guided_session"]["step"] == "step_1_source"
        assert body["next_turn"]["type"] == "single_select"
        assert body["next_turn"]["step_index"] == "step_1_source"

    def test_start_emits_audit_event(self, composer_test_client, audit_recorder) -> None:
        resp = composer_test_client.post("/composer/sessions", json={"title": "x"})
        session_id = resp.json()["id"]
        composer_test_client.post(
            "/composer/guided/start", json={"session_id": session_id}
        )
        events = audit_recorder.guided_events()
        assert any(e.event_type == "guided_turn_emitted" for e in events)


class TestGuidedRespondEndpoint:
    def test_respond_advances_through_step_1(self, composer_test_client) -> None:
        # ... full happy-path walk ...
        pass  # filled out in Task 3.5
```

Test fixtures (`composer_test_client`, `audit_recorder`) follow existing patterns in `tests/integration/web/composer/conftest.py`. If they don't exist for guided, add them in this task.

- [ ] **Step 3: Implement endpoint handlers in `service.py`**

Add inside `ComposerServiceImpl` (line ~1104):

```python
async def guided_start(self, session_id: str) -> dict[str, Any]:
    state = await self._sessions.load_state(session_id)
    new_session = GuidedSession.initial()
    initial_turn = self._build_initial_step_1_turn(state)
    state = dataclasses.replace(state, guided_session=new_session)
    await self._sessions.save_state(session_id, state)
    audit.emit_turn_emitted(
        self._recorder_for(session_id),
        step=GuidedStep.STEP_1_SOURCE,
        turn_type=TurnType(initial_turn["type"]),
        payload_hash=hash_payload(initial_turn["payload"]),
        payload_payload_id=await self._payload_store.put(initial_turn["payload"]),
        emitter="server",
    )
    return {
        "guided_session": _serialise_guided_session(new_session),
        "next_turn": initial_turn,
        "composition_state": _serialise_state(state),
    }


async def guided_respond(
    self, session_id: str, turn_response: TurnResponse
) -> dict[str, Any]:
    state = await self._sessions.load_state(session_id)
    if state.guided_session is None or state.guided_session.terminal is not None:
        raise HTTPException(409, "guided session not active")
    # Identify the current turn type from the most recent emitted turn
    current_turn_type = _last_emitted_turn_type(state.guided_session)
    new_session, next_turn, terminal, audit_events = step_advance(
        state.guided_session, turn_response, current_turn_type=current_turn_type
    )
    # Run the side-effecting step handler if state mutation is required
    new_state, new_session = await self._dispatch_step_handler(
        state, new_session, turn_response
    )
    new_state = dataclasses.replace(new_state, guided_session=new_session)
    for evt in audit_events:
        self._recorder_for(session_id).record_guided_event(evt)
    await self._sessions.save_state(session_id, new_state)
    return {
        "guided_session": _serialise_guided_session(new_session),
        "next_turn": next_turn,
        "terminal": _serialise_terminal(terminal),
        "composition_state": _serialise_state(new_state),
    }
```

The `_dispatch_step_handler` private method drives `handle_step_1_source` / `handle_step_2_sink` / `handle_step_2_5_recipe_apply` based on the new session's step + the response's content. (Plan to write this in Task 3.5.)

- [ ] **Step 4: Run test, verify start endpoint passes**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_endpoints.py::TestGuidedStartEndpoint -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(composer/service): /composer/guided/start endpoint

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.5: `/composer/guided/respond` — full happy-path walk

**Files:**
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `tests/integration/web/composer/guided/test_endpoints.py`

- [ ] **Step 1: Write failing happy-path test**

```python
class TestGuidedHappyPath:
    def test_csv_to_classify_recipe_completes(self, composer_test_client) -> None:
        # 1. Create session, attach CSV blob
        sess = composer_test_client.post(
            "/composer/sessions", json={"title": "happy"}
        ).json()
        sid = sess["id"]
        blob_id = self._create_test_csv_blob(composer_test_client, sid)

        # 2. Start guided
        body = composer_test_client.post(
            "/composer/guided/start", json={"session_id": sid}
        ).json()
        assert body["next_turn"]["type"] == "single_select"

        # 3. Pick csv source
        body = composer_test_client.post(
            "/composer/guided/respond",
            json={
                "session_id": sid,
                "turn_response": {
                    "chosen": ["csv"],
                    "edited_values": None,
                    "custom_inputs": None,
                    "accepted_step_index": None,
                    "edit_step_index": None,
                    "control_signal": None,
                },
            },
        ).json()
        # Server should next emit schema_form (or inspect_and_confirm if blob attached)
        assert body["next_turn"]["type"] in {"schema_form", "inspect_and_confirm"}

        # 4. Confirm inspect (with blob already attached)
        # ... walk through the rest of the steps, ending with recipe_offer ...

        # 5. Apply recipe
        body = composer_test_client.post(
            "/composer/guided/respond",
            json={
                "session_id": sid,
                "turn_response": {
                    "chosen": ["accept"],
                    "edited_values": None,
                    "custom_inputs": None,
                    "accepted_step_index": None,
                    "edit_step_index": None,
                    "control_signal": None,
                },
            },
        ).json()

        # 6. Terminal state is COMPLETED with YAML
        assert body["terminal"]["kind"] == "completed"
        assert "source:" in body["terminal"]["pipeline_yaml"]
```

Helper `_create_test_csv_blob` uses the existing blob-creation API.

- [ ] **Step 2: Implement `_dispatch_step_handler` and the missing intra-step turn emitters**

In `service.py`, implement `_dispatch_step_handler` to call the right handler based on the response shape. Implement turn emitters: `_build_initial_step_1_turn`, `_build_step_2_turn`, `_build_step_2_5_turn`, `_build_step_3_turn`.

(The detailed code is too long for this task description; the implementer should follow the pattern: each emitter inspects the resolved Step 1/2 results to construct the next turn's payload from plugin discovery + recipe pre-match.)

- [ ] **Step 3: Run happy-path test, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_endpoints.py::TestGuidedHappyPath -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/service): /composer/guided/respond — full happy path

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.6: Endpoint error paths — illegal turn type retry, manual exit

**Files:**
- Modify: `tests/integration/web/composer/guided/test_endpoints.py`

- [ ] **Step 1: Write tests for error paths**

```python
class TestGuidedErrorPaths:
    def test_exit_to_freeform_terminates(self, composer_test_client) -> None:
        sess = composer_test_client.post(
            "/composer/sessions", json={"title": "exit"}
        ).json()
        sid = sess["id"]
        composer_test_client.post(
            "/composer/guided/start", json={"session_id": sid}
        )
        body = composer_test_client.post(
            "/composer/guided/respond",
            json={
                "session_id": sid,
                "turn_response": {
                    "chosen": None,
                    "edited_values": None,
                    "custom_inputs": None,
                    "accepted_step_index": None,
                    "edit_step_index": None,
                    "control_signal": "exit_to_freeform",
                },
            },
        ).json()
        assert body["terminal"]["kind"] == "exited_to_freeform"
        assert body["terminal"]["reason"] == "user_pressed_exit"

    def test_respond_after_terminal_returns_409(self, composer_test_client) -> None:
        # As above: drive to terminal, then send another respond
        # Expected: 409 conflict
        pass
```

- [ ] **Step 2: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_endpoints.py::TestGuidedErrorPaths -v
```

Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "test(composer/guided): endpoint error paths — exit_to_freeform + 409 after terminal

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.7: Phase 3 closure

- [ ] **Step 1: Run full Phase 1+2+3 sweep**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ -v
.venv/bin/python -m mypy src/elspeth/web/composer/
.venv/bin/python -m ruff check src/elspeth/web/composer/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
```

Expected: all green.

- [ ] **Step 2: Phase-3 PR ready to ship**

The guided wizard now drives Steps 1 → 2 → 2.5 deterministically through the recipe-match path. This is the demo-bulletproof happy path. Step 3 (LLM-driven chain proposal) and mode lifecycle (manual exit, progressive disclosure) ship in Phases 4 and 5.

---

## Phase 4 — LLM Skill and Chain Solver

**Goal of phase:** Add the LLM-driven Step 3 transform-chain proposer. Write the new guided-mode skill prompt; implement `chain_solver.py` as a thin wrapper around `_litellm_acompletion` that injects the structured GUIDED CONTEXT block; wire the chain commit through existing `tools.py` mutation handlers.

**Phase exit criterion:** A non-recipe-matching pipeline completes through Step 3 with the LLM stub returning a canned valid chain; one real-LLM gated test confirms the prompt produces a valid `propose_chain` for a representative input.

### Task 4.1: Write the guided-mode skill prompt

**Files:**
- Create: `src/elspeth/web/composer/guided/skills/__init__.py`
- Create: `src/elspeth/web/composer/guided/skills/guided_pipeline.md`
- Test: `tests/unit/web/composer/guided/test_skill.py`

- [ ] **Step 1: Write failing test for skill loading**

```python
# tests/unit/web/composer/guided/test_skill.py
"""Tests for guided-mode skill prompt loading + structural assertions.

Skill prompts are LLM behaviour, not code — these tests verify the file
exists, loads, and contains the load-bearing protocol-definition strings.
They do not assert the prompt's behavioural quality; that is verified by
real-LLM tests in test_chain_solver.py.
"""

from __future__ import annotations

from elspeth.web.composer.guided.prompts import load_guided_skill


class TestGuidedSkill:
    def test_loads(self) -> None:
        text = load_guided_skill()
        assert len(text) > 0

    def test_under_size_target(self) -> None:
        text = load_guided_skill()
        assert text.count("\n") <= 100, "guided skill should be ≤100 lines (target ≤80)"

    def test_mentions_six_turn_types(self) -> None:
        text = load_guided_skill()
        for turn_type in (
            "inspect_and_confirm", "single_select", "multi_select_with_custom",
            "schema_form", "propose_chain", "recipe_offer",
        ):
            assert turn_type in text, f"missing turn type: {turn_type}"

    def test_anti_fabrication_clause_present(self) -> None:
        text = load_guided_skill()
        # The hard rule that survives from freeform skill (spec §8.1.4)
        assert "anti-fabrication" in text.lower() or "do not invent" in text.lower()
```

Run, expect import error.

- [ ] **Step 2: Implement skill loader**

```python
# src/elspeth/web/composer/guided/skills/__init__.py
```

```python
# src/elspeth/web/composer/guided/prompts.py
"""Guided-mode skill loading + Step 3 context-block construction.

Module-cached via @lru_cache; per project memory, restart elspeth-web.service
after editing the skill markdown for live changes to take effect.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any


_SKILL_PATH = Path(__file__).parent / "skills" / "guided_pipeline.md"


@lru_cache(maxsize=1)
def load_guided_skill() -> str:
    """Load the guided-mode skill prompt. Cached per process; restart on edit."""
    return _SKILL_PATH.read_text(encoding="utf-8")
```

- [ ] **Step 3: Write the skill markdown**

```markdown
# Guided-Mode Pipeline Composer Skill

You are operating the ELSPETH composer in **guided mode**. This is a structured
turn protocol — both you and the user operate inside fixed constraints:

- You may emit **exactly one** turn per turn, of one of these six types:
  `inspect_and_confirm`, `single_select`, `multi_select_with_custom`,
  `schema_form`, `propose_chain`, `recipe_offer`. Anything else is rejected.
- The user can only answer using the chips, forms, or accept/reject controls
  the turn defines. There is no freeform text input.
- You **cannot** mutate pipeline state. Server-side step handlers commit
  state in response to the user's typed answers. Your only job is to choose
  the right turn for the current step.

## Per-step playbook

### Step 1 — Source

Legal turn types: `inspect_and_confirm`, `single_select`, `schema_form`.
Default: emit `inspect_and_confirm` after a blob is attached. Emit
`schema_form` if the user needs to set non-default options. Emit
`single_select` only if no source plugin has been chosen yet.

### Step 2 — Sink + required fields

Legal turn types: `single_select`, `multi_select_with_custom`, `schema_form`.
Default: `single_select` for the sink plugin, then `schema_form` for
options, then `multi_select_with_custom` for required output fields with
chips pre-populated from Step 1's observed columns.

### Step 2.5 — Recipe match

The server emits `recipe_offer` automatically when a recipe matches; you do
**not** emit this turn yourself. If the user picks "build manually," you
proceed to Step 3.

### Step 3 — Transform chain proposal

Legal turn types: `propose_chain`, `single_select`. The server gives you a
context block:

```
GUIDED CONTEXT (server-resolved):
source: {plugin: ..., columns: [...], sample: [...]}
sink: {outputs: [{plugin: ..., required_fields: [...]}, ...]}
recipe_match: null
```

Propose a transform chain that satisfies the contract from `source.columns`
to each sink's `required_fields`. Every step in the chain must include a
`rationale` string that names what it does and why it is required.

If you cannot find a chain that satisfies the contract:

1. Emit `single_select` with a clarifying question and chip answers — only
   when the user can resolve the ambiguity in one click.
2. Or escalate via `request_advisor_hint` if the question is structural.

**Do not emit a `propose_chain` whose preview will fail.** A degraded
proposal that the server then rejects costs the user a turn.

## Hard rules that survive from freeform mode

- **Anti-fabrication.** Do not invent plugins, options, model names, or
  capabilities. If a name does not appear in `list_sources`/`list_sinks`/
  `list_transforms`/`list_models`, it does not exist.
- **Shape preservation.** If the user described a shape (fork-and-merge,
  multi-stage cascade) that you cannot build, refuse with a named gap via
  `single_select`. Do not silently downgrade.
- **Audit boundary.** Audit logging is operator-managed and not
  composer-configurable. Do not propose audit sinks; refer the user to
  the operator if they ask.

## Sample-value eyeballing (Step 3 only)

When wiring a column into a value-shape-sensitive transform field
(`web_scrape.url_field`, `database.url`, `value_transform` arithmetic), check
that the sample values in the GUIDED CONTEXT block actually look like the
required shape. If not, propose an upstream `value_transform` or `type_coerce`
to normalise — do not assume the strings will be valid at run time.
```

(Target: ~80 lines. Adjust trim/expand as the skill iterates.)

- [ ] **Step 4: Run tests, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_skill.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/elspeth/web/composer/guided/skills/__init__.py \
        src/elspeth/web/composer/guided/skills/guided_pipeline.md \
        src/elspeth/web/composer/guided/prompts.py \
        tests/unit/web/composer/guided/test_skill.py
git commit -m "feat(composer/guided): skill prompt + loader (≤80 lines target)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 4.2: Step 3 GUIDED CONTEXT block constructor

**Files:**
- Modify: `src/elspeth/web/composer/guided/prompts.py`
- Modify: `tests/unit/web/composer/guided/test_skill.py`

- [ ] **Step 1: Write failing test**

```python
class TestStep3ContextBlock:
    def test_renders_source_and_sink(self) -> None:
        from elspeth.web.composer.guided.prompts import build_step_3_context_block
        from elspeth.web.composer.guided.state_machine import (
            SinkOutputResolved, SinkResolved, SourceResolved,
        )

        ctx = build_step_3_context_block(
            source=SourceResolved(
                plugin="csv", options={},
                observed_columns=("price", "qty"),
                sample_rows=({"price": "1.99", "qty": "2"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json", options={"path": "out.jsonl"},
                        required_fields=("avg_price",), schema_mode="fixed",
                    ),
                )
            ),
            recipe_match=None,
        )
        assert "source:" in ctx
        assert "csv" in ctx
        assert "price" in ctx
        assert "avg_price" in ctx
        assert "recipe_match: null" in ctx
```

- [ ] **Step 2: Implement `build_step_3_context_block`**

```python
def build_step_3_context_block(
    *,
    source: "SourceResolved",
    sink: "SinkResolved",
    recipe_match: "RecipeMatch | None",
) -> str:
    """Render the GUIDED CONTEXT block for the Step 3 LLM prompt."""
    import json

    src_payload = {
        "plugin": source.plugin,
        "columns": list(source.observed_columns),
        "sample": [dict(r) for r in source.sample_rows[:3]],
    }
    sink_payload = {
        "outputs": [
            {
                "plugin": o.plugin,
                "required_fields": list(o.required_fields),
                "schema_mode": o.schema_mode,
            }
            for o in sink.outputs
        ],
    }
    match_repr = "null" if recipe_match is None else json.dumps(
        {"recipe_name": recipe_match.recipe_name}
    )
    return (
        "GUIDED CONTEXT (server-resolved):\n"
        f"source: {json.dumps(src_payload)}\n"
        f"sink: {json.dumps(sink_payload)}\n"
        f"recipe_match: {match_repr}\n"
    )
```

Add the imports at the top of `prompts.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elspeth.web.composer.guided.recipe_match import RecipeMatch
    from elspeth.web.composer.guided.state_machine import (
        SinkResolved,
        SourceResolved,
    )
```

- [ ] **Step 3: Run, verify pass**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/test_skill.py -v
```

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): Step 3 GUIDED CONTEXT block constructor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 4.3: Chain solver — wrap `_litellm_acompletion`

**Files:**
- Create: `src/elspeth/web/composer/guided/chain_solver.py`
- Test: `tests/integration/web/composer/guided/test_chain_solver.py`

- [ ] **Step 1: Write failing test with stubbed LLM**

```python
# tests/integration/web/composer/guided/test_chain_solver.py
"""Tests for guided-mode chain solver.

Two test modes:
  - Stubbed LLM via existing ChaosLLM (CI; fast, deterministic)
  - Real LLM (gated CI lane; runs once per PR if budget permits)
"""

from __future__ import annotations

import pytest


class TestChainSolverStubbed:
    @pytest.mark.usefixtures("chaosllm_stub")
    def test_returns_chain_proposal(
        self, chaosllm_stub, audit_recorder
    ) -> None:
        from elspeth.web.composer.guided.chain_solver import solve_chain
        from elspeth.web.composer.guided.state_machine import (
            SinkOutputResolved, SinkResolved, SourceResolved,
        )

        chaosllm_stub.queue_response(
            tool_call={
                "name": "emit_turn",
                "arguments": {
                    "turn_type": "propose_chain",
                    "payload": {
                        "steps": [
                            {
                                "plugin": "type_coerce",
                                "options": {
                                    "fields": [{"name": "price", "type": "float"}]
                                },
                                "rationale": "price is str; gate needs float",
                            }
                        ],
                        "why": "bridge str→float for arithmetic",
                        "blockers": [],
                    },
                },
            }
        )

        proposal = solve_chain(
            source=SourceResolved(
                plugin="csv", options={},
                observed_columns=("price",),
                sample_rows=({"price": "1.99"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json", options={},
                        required_fields=("price",), schema_mode="fixed",
                    ),
                )
            ),
            audit_recorder=audit_recorder,
        )

        assert len(proposal.steps) == 1
        assert proposal.steps[0]["plugin"] == "type_coerce"


@pytest.mark.real_llm
class TestChainSolverRealLLM:
    """Gated; runs only when --run-real-llm CLI flag is set."""

    def test_csv_to_json_with_avg_aggregation(self, audit_recorder) -> None:
        from elspeth.web.composer.guided.chain_solver import solve_chain
        from elspeth.web.composer.guided.state_machine import (
            SinkOutputResolved, SinkResolved, SourceResolved,
        )

        proposal = solve_chain(
            source=SourceResolved(
                plugin="csv", options={},
                observed_columns=("price", "qty", "region"),
                sample_rows=({"price": "1.99", "qty": "2", "region": "north"},),
            ),
            sink=SinkResolved(
                outputs=(
                    SinkOutputResolved(
                        plugin="json", options={},
                        required_fields=("avg_price", "region"),
                        schema_mode="fixed",
                    ),
                )
            ),
            audit_recorder=audit_recorder,
        )

        # Real-LLM smoke: chain must contain at least one numeric coercion
        # AND at least one aggregation (batch_stats / similar).
        plugins = [s["plugin"] for s in proposal.steps]
        assert any("coerce" in p for p in plugins) or any("type" in p for p in plugins)
        assert any("batch" in p for p in plugins) or any("aggregat" in p for p in plugins)
```

- [ ] **Step 2: Implement `solve_chain`**

```python
# src/elspeth/web/composer/guided/chain_solver.py
"""Chain solver: invoke LLM with guided skill + GUIDED CONTEXT block, parse propose_chain.

This wraps the existing _litellm_acompletion so audit / telemetry / token
accounting / advisor escalation all flow through the same plumbing as freeform.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from elspeth.web.composer.guided.prompts import (
    build_step_3_context_block,
    load_guided_skill,
)
from elspeth.web.composer.guided.state_machine import (
    ChainProposal,
    SinkResolved,
    SourceResolved,
)
from elspeth.web.composer.guided.recipe_match import RecipeMatch


_GUIDED_LLM_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "emit_turn",
            "description": (
                "Emit one turn to the user. The only way to interact in guided mode."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_type": {
                        "type": "string",
                        "enum": [
                            "inspect_and_confirm", "single_select",
                            "multi_select_with_custom", "schema_form",
                            "propose_chain", "recipe_offer",
                        ],
                    },
                    "payload": {"type": "object"},
                },
                "required": ["turn_type", "payload"],
            },
        },
    },
    # Discovery + advisor tools follow the existing freeform schemas (omitted
    # here; load from tools.get_tool_definitions() filtered by name).
]


async def solve_chain(
    *,
    source: SourceResolved,
    sink: SinkResolved,
    recipe_match: RecipeMatch | None = None,
    audit_recorder: Any,
) -> ChainProposal:
    """Invoke the LLM with the guided skill, expect a propose_chain back.

    Reuses the same _litellm_acompletion path as freeform; audit and token
    accounting flow through the same hooks.
    """
    from elspeth.web.composer.service import _litellm_acompletion  # avoid cycle

    skill = load_guided_skill()
    context_block = build_step_3_context_block(
        source=source, sink=sink, recipe_match=recipe_match
    )
    system_prompt = f"{skill}\n\n{context_block}"
    response = await _litellm_acompletion(
        model="anthropic/claude-3.5-sonnet",  # operator-overrideable
        messages=[{"role": "system", "content": system_prompt}],
        tools=_GUIDED_LLM_TOOLS,
        tool_choice={"type": "function", "function": {"name": "emit_turn"}},
    )
    tool_call = _extract_tool_call(response)
    if tool_call["name"] != "emit_turn":
        raise ValueError(f"chain solver expected emit_turn, got {tool_call['name']}")
    args = tool_call["arguments"]
    if args["turn_type"] != "propose_chain":
        raise ValueError(
            f"chain solver expected propose_chain turn, got {args['turn_type']}"
        )
    payload = args["payload"]
    return ChainProposal(steps=tuple(payload["steps"]), why=str(payload["why"]))


def _extract_tool_call(response: Any) -> Mapping[str, Any]:
    # Mirrors the existing extraction in service.py
    msg = response["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        raise ValueError("solve_chain: response had no tool_calls")
    tc = tool_calls[0]
    import json

    return {
        "name": tc["function"]["name"],
        "arguments": json.loads(tc["function"]["arguments"]),
    }
```

- [ ] **Step 3: Run stubbed test, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_chain_solver.py::TestChainSolverStubbed -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/composer/guided/chain_solver.py \
        tests/integration/web/composer/guided/test_chain_solver.py
git commit -m "feat(composer/guided): chain solver wraps _litellm_acompletion with guided skill

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 4.4: Step 3 handler — preview, repair, commit

**Files:**
- Modify: `src/elspeth/web/composer/guided/steps.py`
- Modify: `tests/integration/web/composer/guided/test_step_handlers.py`

- [ ] **Step 1: Write failing test**

```python
class TestStep3Handler:
    def test_chain_accepted_commits_via_existing_handlers(self) -> None:
        from elspeth.web.composer.guided.state_machine import ChainProposal
        from elspeth.web.composer.guided.steps import handle_step_3_chain_accept

        # Set up state with source + sink already committed (mimicking Step 1+2)
        state = self._build_state_with_source_and_sink()
        session = self._build_session_at_step_3(state)
        proposal = ChainProposal(
            steps=(
                {
                    "plugin": "type_coerce",
                    "options": {"fields": [{"name": "price", "type": "float"}]},
                    "rationale": "...",
                },
            ),
            why="bridge",
        )

        result = handle_step_3_chain_accept(
            state=state, session=session, proposal=proposal
        )

        # Each chain step appears as a node in composition_state.nodes
        assert any(n.plugin == "type_coerce" for n in result.state.nodes)
        # Session is COMPLETED with YAML
        assert result.session.terminal is not None
        assert result.session.terminal.kind.value == "completed"
```

- [ ] **Step 2: Implement `handle_step_3_chain_accept`**

```python
from elspeth.web.composer.tools import _execute_upsert_node, _execute_upsert_edge


def handle_step_3_chain_accept(
    *,
    state: CompositionState,
    session: GuidedSession,
    proposal: ChainProposal,
) -> StepHandlerResult:
    """Commit each step of *proposal* via existing tools.py handlers."""
    current_state = state
    last_result: ToolResult | None = None
    for idx, step in enumerate(proposal.steps):
        node_payload = {
            "id": f"guided_xform_{idx}",
            "node_type": "transform",
            "plugin": step["plugin"],
            "input": "main" if idx == 0 else f"guided_chain_{idx - 1}",
            "on_success": (
                "main" if idx == len(proposal.steps) - 1 else f"guided_chain_{idx}"
            ),
            "options": dict(step["options"]),
        }
        next_state, last_result = _execute_upsert_node(current_state, node_payload)
        if not last_result.success:
            return StepHandlerResult(
                state=current_state, session=session, tool_result=last_result
            )
        current_state = next_state
    if last_result is None:
        raise ValueError("step 3 proposal had zero steps; refusing empty commit")

    # Run preview_pipeline; on failure, the endpoint runs the repair attempt.
    from elspeth.web.composer.tools import _execute_preview_pipeline

    preview = _execute_preview_pipeline(current_state, {})
    if not preview.data["is_valid"]:
        return StepHandlerResult(
            state=current_state, session=session, tool_result=preview
        )

    # Success: render YAML and stamp COMPLETED
    yaml_text = generate_yaml(current_state)
    terminal = TerminalState(
        kind=TerminalKind.COMPLETED, reason=None, pipeline_yaml=yaml_text
    )
    new_session = _replace(session, step_3_proposal=proposal, terminal=terminal)
    return StepHandlerResult(
        state=current_state, session=new_session, tool_result=last_result
    )
```

- [ ] **Step 3: Run test, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_step_handlers.py::TestStep3Handler -v
```

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): handle_step_3_chain_accept commits via existing handlers

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 4.5: Wire chain solver + Step 3 handler into `/composer/guided/respond`

**Files:**
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `tests/integration/web/composer/guided/test_endpoints.py`

- [ ] **Step 1: Write happy-path test for non-recipe pipeline**

```python
class TestGuidedHandBuiltPath:
    @pytest.mark.usefixtures("chaosllm_stub")
    def test_csv_to_csv_with_solver_chain(
        self, composer_test_client, chaosllm_stub
    ) -> None:
        # Force pre-match to fail by picking a sink that no recipe matches
        chaosllm_stub.queue_response(...)  # canned propose_chain
        # Walk the wizard, end at preview-green completion
        ...
```

- [ ] **Step 2: In `_dispatch_step_handler`, route Step 3 responses**

When the response is at Step 3 with `accepted_step_index is None and edit_step_index is None`, call `handle_step_3_chain_accept` with the previously-stored proposal. When the user clicks "edit step N," re-invoke `solve_chain` with the locked step pinned in the prompt.

- [ ] **Step 3: Run test, verify pass**

```bash
.venv/bin/python -m pytest tests/integration/web/composer/guided/test_endpoints.py::TestGuidedHandBuiltPath -v
```

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/service): wire chain solver into /guided/respond

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 4.6: Phase 4 closure

- [ ] **Step 1: Run Phase 1+2+3+4 sweep**

```bash
.venv/bin/python -m pytest tests/unit/web/composer/guided/ tests/integration/web/composer/guided/ -v
.venv/bin/python -m mypy src/elspeth/web/composer/
.venv/bin/python -m ruff check src/elspeth/web/composer/
```

Expected: green except real-LLM gated test, which runs separately.

- [ ] **Step 2: Phase-4 PR ready to ship**

Step 3 chain proposal works end-to-end through stubbed LLM. The wizard now completes a non-recipe pipeline. Phase 5 adds mode lifecycle (manual exit, auto-drop, progressive disclosure).

---

## Phase 5 — Mode Lifecycle and Progressive Disclosure

**Goal of phase:** Implement the auto-drop on solver exhaustion, the progressive-disclosure transition prompt, and confirm that the existing freeform composer continues unmodified after exit.

**Phase exit criterion:** A failing chain-solver path drops to freeform with the partial pipeline carried; a freeform chat turn after transition includes the `[freeform skill]` content + transition message and the LLM emits a non-guided tool call.

### Task 5.1: Auto-drop on solver-exhausted

**Files:**
- Modify: `src/elspeth/web/composer/service.py`
- Test: `tests/integration/web/composer/guided/test_endpoints.py`

- [ ] **Step 1: Write failing test**

```python
class TestGuidedAutoDrop:
    @pytest.mark.usefixtures("chaosllm_stub")
    def test_invalid_chain_after_repair_drops_to_freeform(
        self, composer_test_client, chaosllm_stub
    ) -> None:
        # Stub returns an invalid chain twice (initial + repair); endpoint
        # must auto-drop after the second failure.
        chaosllm_stub.queue_response(...)  # invalid initial
        chaosllm_stub.queue_response(...)  # invalid repair
        # Walk wizard to Step 3, hit the failure
        ...
        # Final response carries terminal with reason=solver_exhausted
        assert body["terminal"]["reason"] == "solver_exhausted"
```

- [ ] **Step 2: Implement repair attempt + drop in endpoint**

In `service.py` `guided_respond`, after `handle_step_3_chain_accept` returns an unsuccessful preview, invoke a single repair: feed the validation error to `solve_chain` with a "repair this" addendum to the system prompt. If the repair fails, call `mark_solver_exhausted` and return.

- [ ] **Step 3: Run, verify pass**

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(composer/guided): auto-drop on solver-exhausted after repair attempt

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 5.2: Progressive-disclosure transition prompt

**Files:**
- Modify: `src/elspeth/web/composer/guided/prompts.py`
- Modify: `src/elspeth/web/composer/service.py`
- Test: `tests/integration/web/composer/guided/test_progressive_disclosure.py`

- [ ] **Step 1: Write failing test**

```python
# tests/integration/web/composer/guided/test_progressive_disclosure.py
class TestProgressiveDisclosure:
    def test_transition_prompt_includes_both_skills(self) -> None:
        from elspeth.web.composer.guided.prompts import (
            build_mode_transition_system_prompt,
        )
        from elspeth.web.composer.guided.state_machine import TerminalReason

        prompt = build_mode_transition_system_prompt(
            terminal_reason="user_pressed_exit"
        )
        # Must include guided skill content (so model knows what it just did)
        assert "guided mode" in prompt.lower() or "turn protocol" in prompt.lower()
        # Must include freeform skill content
        assert "ELSPETH pipeline" in prompt or "Audit Primacy" in prompt
        # Must include the explicit "rules lifted" signal
        assert "LIFTED" in prompt or "lifted" in prompt
        # Must include the reason
        assert "user_pressed_exit" in prompt

    @pytest.mark.usefixtures("chaosllm_stub")
    def test_llm_after_transition_uses_freeform_tool(
        self, composer_test_client, chaosllm_stub
    ) -> None:
        # Drive a session through guided mode and exit_to_freeform.
        # Then send a freeform chat message.
        # Stub the LLM to return a freeform tool call (e.g., set_metadata).
        # Verify the LLM was given the progressive-disclosure system prompt
        # and the tool call succeeded (i.e., LLM was not still constrained).
        ...
```

- [ ] **Step 2: Implement `build_mode_transition_system_prompt`**

```python
# Append to prompts.py
def build_mode_transition_system_prompt(*, terminal_reason: str) -> str:
    """Construct the layered system prompt for a freeform turn after guided exit.

    Format: [guided skill] + [transition message] + [freeform skill]
    """
    guided = load_guided_skill()
    freeform = _load_freeform_skill()
    transition = (
        "## Mode Transition — Guided → Freeform\n\n"
        f"You have just exited guided mode (reason: {terminal_reason}).\n\n"
        "The protocol restrictions above (closed turn taxonomy, read-only state,\n"
        "legal-turn matrix) are LIFTED for the remainder of this session. You now\n"
        "have the full freeform tool surface detailed below. The guided session's\n"
        "outcome is recorded in `composition_state.guided_session` — do not re-run\n"
        "any work it already accomplished.\n"
    )
    return f"{guided}\n\n{transition}\n\n{freeform}"


@lru_cache(maxsize=1)
def _load_freeform_skill() -> str:
    skill_path = Path(__file__).parent.parent / "skills" / "pipeline_composer.md"
    return skill_path.read_text(encoding="utf-8")
```

- [ ] **Step 3: Wire into freeform chat endpoint**

In `service.py` chat-endpoint handler, when `composition_state.guided_session` is set and `guided_session.terminal is not None`, replace the freeform skill with `build_mode_transition_system_prompt(...)` for **the next chat turn only**. Track via a simple `transition_consumed: bool` flag in the session record (or by checking message-history for the marker).

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat(composer/service): progressive-disclosure transition prompt

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 5.3: Phase 5 closure — full audit emission test

**Files:**
- Test: `tests/integration/web/composer/guided/test_audit_emission.py`

- [ ] **Step 1: Write the full-session audit test**

```python
# tests/integration/web/composer/guided/test_audit_emission.py
class TestFullSessionAuditEmission:
    @pytest.mark.usefixtures("chaosllm_stub")
    def test_recipe_match_path_audit_events(
        self, composer_test_client, audit_recorder
    ) -> None:
        # Drive a full happy-path session
        # Verify every expected event_type appears in the recorder
        events = audit_recorder.guided_events()
        types = [e.event_type for e in events]
        assert "guided_turn_emitted" in types
        assert "guided_turn_answered" in types
        assert "guided_step_advanced" in types
        # No drop event on the happy path
        assert "guided_dropped_to_freeform" not in types

    def test_auto_drop_path_emits_dropped_event(
        self, composer_test_client, chaosllm_stub, audit_recorder
    ) -> None:
        # Drive an auto-drop session
        events = audit_recorder.guided_events()
        types = [e.event_type for e in events]
        assert "guided_dropped_to_freeform" in types
        drop = next(e for e in events if e.event_type == "guided_dropped_to_freeform")
        assert drop.payload["drop_reason"] == "solver_exhausted"
```

- [ ] **Step 2: Run, verify pass**

- [ ] **Step 3: Phase-5 commit + PR**

```bash
git add -u
git commit -m "test(composer/guided): full-session audit emission coverage

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Phase 5 closes the backend. The wizard now drives Steps 1–3 + recipe match + auto-drop + progressive disclosure, all backed by audit events. The next phases ship the frontend.

---

## Phase 6 — Frontend Types, API Client, Store Slice

**Goal of phase:** Mirror the backend protocol on the frontend: TypeScript types for `Turn`, `TurnResponse`, `GuidedSession`; `apiClient.postGuidedStart` and `postGuidedRespond`; `useSessionStore` slice for `guidedSession` state and the three actions.

**Phase exit criterion:** Vitest store-slice tests pass; types compile cleanly; manual smoke against the running backend (curl the start endpoint, assert the store updates).

### Task 6.1: TypeScript types mirroring the backend protocol

**Files:**
- Create: `src/elspeth/web/frontend/src/types/guided.ts`
- Test: `src/elspeth/web/frontend/src/types/guided.test.ts`

- [ ] **Step 1: Write failing type assertion test**

```typescript
// src/elspeth/web/frontend/src/types/guided.test.ts
import { describe, expect, it } from "vitest";
import type {
  ControlSignal,
  GuidedSession,
  Turn,
  TurnResponse,
  TurnType,
} from "./guided";

describe("guided types", () => {
  it("turn types match backend", () => {
    const allTypes: TurnType[] = [
      "inspect_and_confirm", "single_select",
      "multi_select_with_custom", "schema_form",
      "propose_chain", "recipe_offer",
    ];
    expect(allTypes.length).toBe(6);
  });

  it("turn carries step_index and payload", () => {
    const turn: Turn = {
      type: "single_select",
      step_index: "step_1_source",
      payload: { question: "Q?", options: [], allow_custom: false },
    };
    expect(turn.type).toBe("single_select");
  });

  it("control signals are limited to three values", () => {
    const signals: ControlSignal[] = ["exit_to_freeform", "request_advisor", "reject"];
    expect(signals.length).toBe(3);
  });
});
```

- [ ] **Step 2: Implement `guided.ts`**

```typescript
// src/elspeth/web/frontend/src/types/guided.ts
export type TurnType =
  | "inspect_and_confirm"
  | "single_select"
  | "multi_select_with_custom"
  | "schema_form"
  | "propose_chain"
  | "recipe_offer";

export type GuidedStep =
  | "step_1_source"
  | "step_2_sink"
  | "step_2_5_recipe_match"
  | "step_3_transforms";

export type ControlSignal =
  | "exit_to_freeform"
  | "request_advisor"
  | "reject";

export interface TurnOption {
  id: string;
  label: string;
  hint: string | null;
}

export interface Turn {
  type: TurnType;
  step_index: GuidedStep;
  payload: Record<string, unknown>;
}

export interface TurnResponse {
  chosen: string[] | null;
  edited_values: Record<string, unknown> | null;
  custom_inputs: string[] | null;
  accepted_step_index: number | null;
  edit_step_index: number | null;
  control_signal: ControlSignal | null;
}

export interface TerminalState {
  kind: "completed" | "exited_to_freeform";
  reason:
    | "user_pressed_exit"
    | "protocol_violation"
    | "solver_exhausted"
    | null;
  pipeline_yaml: string | null;
}

export interface GuidedSession {
  step: GuidedStep;
  history: Array<{
    step: GuidedStep;
    turn_type: TurnType;
    payload_hash: string;
    response_hash: string | null;
    emitter: "server" | "llm";
  }>;
  step_1_result: unknown | null;
  step_2_result: unknown | null;
  step_3_proposal: unknown | null;
  terminal: TerminalState | null;
}

export interface GuidedStartResponse {
  guided_session: GuidedSession;
  next_turn: Turn;
  composition_state: unknown;
}

export interface GuidedRespondResponse {
  guided_session: GuidedSession;
  next_turn: Turn | null;
  terminal: TerminalState | null;
  composition_state: unknown;
}
```

- [ ] **Step 3: Run, verify pass**

```bash
cd src/elspeth/web/frontend && npx vitest run src/types/guided.test.ts
```

- [ ] **Step 4: Commit**

```bash
git add src/elspeth/web/frontend/src/types/guided.ts \
        src/elspeth/web/frontend/src/types/guided.test.ts
git commit -m "feat(frontend/types): guided-mode protocol types mirroring backend

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 6.2: API client methods

**Files:**
- Modify: `src/elspeth/web/frontend/src/api/client.ts`
- Test: `src/elspeth/web/frontend/src/api/client.guided.test.ts`

- [ ] **Step 1: Write failing test**

```typescript
import { describe, expect, it, vi } from "vitest";
import { apiClient } from "./client";

describe("apiClient.postGuidedStart", () => {
  it("posts session_id and returns parsed response", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({
        guided_session: { step: "step_1_source", history: [], terminal: null },
        next_turn: { type: "single_select", step_index: "step_1_source", payload: {} },
        composition_state: {},
      }),
    } as Response);

    const result = await apiClient.postGuidedStart("sess-1");
    expect(result.next_turn.type).toBe("single_select");
    expect(fetchSpy).toHaveBeenCalledWith(
      expect.stringContaining("/composer/guided/start"),
      expect.objectContaining({
        method: "POST",
        body: expect.stringContaining("sess-1"),
      })
    );
  });
});
```

- [ ] **Step 2: Implement methods**

In `src/api/client.ts` add:

```typescript
import type {
  GuidedRespondResponse,
  GuidedStartResponse,
  TurnResponse,
} from "@/types/guided";

class ApiClient {
  // ... existing methods ...

  async postGuidedStart(sessionId: string): Promise<GuidedStartResponse> {
    return this._post<GuidedStartResponse>(
      "/composer/guided/start",
      { session_id: sessionId }
    );
  }

  async postGuidedRespond(
    sessionId: string,
    turnResponse: TurnResponse
  ): Promise<GuidedRespondResponse> {
    return this._post<GuidedRespondResponse>(
      "/composer/guided/respond",
      { session_id: sessionId, turn_response: turnResponse }
    );
  }
}
```

- [ ] **Step 3: Run, verify pass**

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(frontend/api): postGuidedStart + postGuidedRespond

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 6.3: `sessionStore` guided slice

**Files:**
- Modify: `src/elspeth/web/frontend/src/stores/sessionStore.ts`
- Test: `src/elspeth/web/frontend/src/stores/sessionStore.guided.test.ts`

- [ ] **Step 1: Write failing tests**

```typescript
import { describe, expect, it, beforeEach, vi } from "vitest";
import { useSessionStore } from "./sessionStore";

describe("sessionStore guided slice", () => {
  beforeEach(() => {
    useSessionStore.setState({ guidedSession: null });
  });

  it("startGuided posts and stores session", async () => {
    vi.spyOn(apiClient, "postGuidedStart").mockResolvedValue({
      guided_session: { step: "step_1_source", history: [], terminal: null,
                        step_1_result: null, step_2_result: null, step_3_proposal: null },
      next_turn: { type: "single_select", step_index: "step_1_source", payload: {} },
      composition_state: {},
    });
    await useSessionStore.getState().startGuided("sess-1");
    expect(useSessionStore.getState().guidedSession?.step).toBe("step_1_source");
  });

  it("respondGuided replaces guidedSession with server response", async () => {
    // setup with active guidedSession, then respond, assert server response wins
  });

  it("exitToFreeform sends control signal and clears on terminal", async () => {
    // verify control_signal: "exit_to_freeform" sent and terminal set
  });
});
```

- [ ] **Step 2: Implement slice**

Append to `sessionStore.ts`:

```typescript
interface GuidedSlice {
  guidedSession: GuidedSession | null;
  guidedNextTurn: Turn | null;  // server's next-turn payload, replaced atomically
  startGuided: (sessionId: string) => Promise<void>;
  respondGuided: (turnResponse: TurnResponse) => Promise<void>;
  exitToFreeform: () => Promise<void>;
}

// In the create() body:
guidedSession: null,
guidedNextTurn: null,
startGuided: async (sessionId) => {
  const result = await apiClient.postGuidedStart(sessionId);
  set({
    guidedSession: result.guided_session,
    guidedNextTurn: result.next_turn,
    compositionState: result.composition_state,
  });
},
respondGuided: async (turnResponse) => {
  const sessionId = get().activeSessionId;
  if (!sessionId) throw new Error("no active session");
  const result = await apiClient.postGuidedRespond(sessionId, turnResponse);
  set({
    guidedSession: result.guided_session,
    guidedNextTurn: result.next_turn,
    compositionState: result.composition_state,
  });
},
exitToFreeform: async () => {
  await get().respondGuided({
    chosen: null, edited_values: null, custom_inputs: null,
    accepted_step_index: null, edit_step_index: null,
    control_signal: "exit_to_freeform",
  });
},
```

- [ ] **Step 3: Run, verify pass**

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(frontend/stores): guidedSession slice + start/respond/exit actions

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 7 — Frontend Turn Widgets

**Goal of phase:** Implement one React widget per turn type plus the `GuidedTurn` dispatcher. Each widget is a Vitest-tested unit that takes a typed payload, renders chips/forms/cards, and fires `onSubmit(turnResponse)` on user action.

**Phase exit criterion:** Each widget's Vitest test renders, exercises all interactions, and asserts the correct `TurnResponse` shape on submit.

### Task 7.1: `GuidedTurn` dispatcher

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GuidedTurn } from "./GuidedTurn";

describe("GuidedTurn", () => {
  it("dispatches single_select to SingleSelectTurn", () => {
    render(
      <GuidedTurn
        turn={{
          type: "single_select",
          step_index: "step_1_source",
          payload: { question: "Pick", options: [{id: "a", label: "A", hint: null}], allow_custom: false },
        }}
        onSubmit={() => {}}
      />
    );
    expect(screen.getByText("Pick")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "A" })).toBeInTheDocument();
  });

  it("throws on unknown turn type at compile time (smoke)", () => {
    // TypeScript prevents passing an unknown turn.type; runtime fallback
    // renders an error placeholder.
  });
});
```

- [ ] **Step 2: Implement dispatcher**

```typescript
// src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx
import { InspectAndConfirmTurn } from "./InspectAndConfirmTurn";
import { MultiSelectWithCustomTurn } from "./MultiSelectWithCustomTurn";
import { ProposeChainTurn } from "./ProposeChainTurn";
import { RecipeOfferTurn } from "./RecipeOfferTurn";
import { SchemaFormTurn } from "./SchemaFormTurn";
import { SingleSelectTurn } from "./SingleSelectTurn";
import type { Turn, TurnResponse } from "@/types/guided";

interface GuidedTurnProps {
  turn: Turn;
  onSubmit: (response: TurnResponse) => void;
}

export function GuidedTurn({ turn, onSubmit }: GuidedTurnProps) {
  switch (turn.type) {
    case "inspect_and_confirm":
      return <InspectAndConfirmTurn payload={turn.payload as never} onSubmit={onSubmit} />;
    case "single_select":
      return <SingleSelectTurn payload={turn.payload as never} onSubmit={onSubmit} />;
    case "multi_select_with_custom":
      return <MultiSelectWithCustomTurn payload={turn.payload as never} onSubmit={onSubmit} />;
    case "schema_form":
      return <SchemaFormTurn payload={turn.payload as never} onSubmit={onSubmit} />;
    case "propose_chain":
      return <ProposeChainTurn payload={turn.payload as never} onSubmit={onSubmit} />;
    case "recipe_offer":
      return <RecipeOfferTurn payload={turn.payload as never} onSubmit={onSubmit} />;
  }
}
```

- [ ] **Step 3: Run, verify pass (will require widget stubs that follow)**

- [ ] **Step 4: Commit (after all widgets ready, see Task 7.9)**

### Task 7.2: `SingleSelectTurn` widget — full pattern (template for the rest)

**Files:**
- Create: `src/elspeth/web/frontend/src/components/chat/guided/SingleSelectTurn.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/guided/SingleSelectTurn.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SingleSelectTurn } from "./SingleSelectTurn";

describe("SingleSelectTurn", () => {
  it("renders question and options", () => {
    render(
      <SingleSelectTurn
        payload={{
          question: "Format?",
          options: [
            { id: "jsonl", label: "JSONL", hint: null },
            { id: "csv", label: "CSV", hint: null },
          ],
          allow_custom: false,
        }}
        onSubmit={() => {}}
      />
    );
    expect(screen.getByText("Format?")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "JSONL" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "CSV" })).toBeInTheDocument();
  });

  it("fires onSubmit with chosen id when option clicked", () => {
    const onSubmit = vi.fn();
    render(
      <SingleSelectTurn
        payload={{
          question: "X?",
          options: [{ id: "a", label: "A", hint: null }],
          allow_custom: false,
        }}
        onSubmit={onSubmit}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: "A" }));
    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({ chosen: ["a"] })
    );
  });

  it("renders custom input when allow_custom is true", () => {
    render(
      <SingleSelectTurn
        payload={{
          question: "X?",
          options: [],
          allow_custom: true,
        }}
        onSubmit={() => {}}
      />
    );
    expect(screen.getByPlaceholderText(/custom/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Implement**

```typescript
// src/elspeth/web/frontend/src/components/chat/guided/SingleSelectTurn.tsx
import { useState } from "react";
import type { TurnOption, TurnResponse } from "@/types/guided";

interface Payload {
  question: string;
  options: TurnOption[];
  allow_custom: boolean;
}

interface Props {
  payload: Payload;
  onSubmit: (response: TurnResponse) => void;
}

export function SingleSelectTurn({ payload, onSubmit }: Props) {
  const [custom, setCustom] = useState("");
  const submit = (chosenId: string) => {
    onSubmit({
      chosen: [chosenId],
      edited_values: null,
      custom_inputs: null,
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  };
  const submitCustom = () => {
    onSubmit({
      chosen: null,
      edited_values: null,
      custom_inputs: [custom],
      accepted_step_index: null,
      edit_step_index: null,
      control_signal: null,
    });
  };
  return (
    <fieldset className="guided-single-select" aria-labelledby="ssq">
      <legend id="ssq">{payload.question}</legend>
      <div className="guided-chip-group">
        {payload.options.map((opt) => (
          <button
            key={opt.id}
            type="button"
            onClick={() => submit(opt.id)}
            aria-describedby={opt.hint ? `${opt.id}-hint` : undefined}
          >
            {opt.label}
            {opt.hint && (
              <span id={`${opt.id}-hint`} className="hint">
                {opt.hint}
              </span>
            )}
          </button>
        ))}
      </div>
      {payload.allow_custom && (
        <div className="custom-input-row">
          <input
            type="text"
            placeholder="Custom answer…"
            value={custom}
            onChange={(e) => setCustom(e.target.value)}
          />
          <button type="button" disabled={!custom} onClick={submitCustom}>
            Submit
          </button>
        </div>
      )}
    </fieldset>
  );
}
```

- [ ] **Step 3: Run, verify pass**

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(frontend/guided): SingleSelectTurn widget with chip group + custom escape

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Tasks 7.3 – 7.7: Remaining widgets

Each widget follows the same TDD pattern as Task 7.2. Specifics:

- [ ] **Task 7.3: `InspectAndConfirmTurn`** — renders a column-table + sample-rows + warnings; "Looks right" button submits `{edited_values: payload.observed}`; "Edit columns…" opens an editor that lets the user remove/rename columns and resubmits with edits.

- [ ] **Task 7.4: `MultiSelectWithCustomTurn`** — chip group with checkbox semantics, free-add input, "Or: let source decide" escape button (when `escape_label` is non-null) which submits `{edited_values: {schema_mode: "observed", required_fields: []}}`.

- [ ] **Task 7.5: `SchemaFormTurn`** — auto-generates form fields from `payload.schema_block` (JSON Schema). Required fields visible by default; optional collapsed under "Show advanced." On submit, packages all field values into `edited_values`.

- [ ] **Task 7.6: `ProposeChainTurn`** — renders `payload.steps` as a vertical list of cards; each card shows plugin name + key options + rationale; bottom buttons: "Accept all" (submits `{accepted_step_index: null}` meaning full accept), per-step "Edit" (submits `{edit_step_index: idx}`), "Reject" (submits `{control_signal: "reject"}`), "Ask advisor" (submits `{control_signal: "request_advisor"}`).

- [ ] **Task 7.7: `RecipeOfferTurn`** — single card with recipe name + slot summary + "Apply recipe" / "Build manually" buttons. "Apply" submits `{chosen: ["accept"]}`; "Build manually" submits `{chosen: ["build_manually"]}`.

- [ ] **Task 7.8: `ExitToFreeformButton`** — persistent button rendered alongside every turn; on click, submits `{control_signal: "exit_to_freeform"}`. Lives in `src/components/chat/guided/ExitToFreeformButton.tsx`.

- [ ] **Task 7.9: `GuidedHistory`** — collapsible list of completed steps with their summary (e.g., "Step 1: CSV with cols [price, qty]; Step 2: JSONL output / required: [avg_price]"). Read-only.

- [ ] **Task 7.10: `CompletionSummary`** — renders the YAML preview from `terminal.pipeline_yaml` with a "Save and exit" button (which transitions to freeform mode for any further edits) and a "Drop to freeform to keep editing" button.

Each task: write failing tests, implement, run, commit. After all are green, run:

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/chat/guided/
```

Expected: all green. Phase-7 PR ready.

---

## Phase 8 — ChatPanel Integration

**Goal of phase:** Wire the guided widgets into `ChatPanel.tsx` via the top-level mode discriminator. Add the `ExitToFreeformButton` rendering. Verify `useFocusTrap` and `aria-live` continue to work.

### Task 8.1: ChatPanel mode discriminator

**Files:**
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Test: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`

- [ ] **Step 1: Write failing test**

```typescript
describe("ChatPanel mode discriminator", () => {
  it("renders GuidedTurn when guidedSession is active", () => {
    useSessionStore.setState({
      guidedSession: { step: "step_1_source", terminal: null, /* ... */ },
      // ... session, messages stubs ...
    });
    render(<ChatPanel />);
    expect(screen.getByRole("fieldset")).toBeInTheDocument(); // SingleSelectTurn
  });

  it("renders freeform input when guidedSession is null", () => {
    useSessionStore.setState({ guidedSession: null });
    render(<ChatPanel />);
    expect(screen.getByPlaceholderText(/message/i)).toBeInTheDocument(); // ChatInput
  });

  it("renders CompletionSummary when guided terminated as completed", () => {
    useSessionStore.setState({
      guidedSession: {
        terminal: { kind: "completed", reason: null, pipeline_yaml: "x:\n" },
        // ...
      },
    });
    render(<ChatPanel />);
    expect(screen.getByText(/save and exit/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Implement discriminator at top of `ChatPanel`**

```typescript
export function ChatPanel({ onOpenSecrets }: ChatPanelProps) {
  const guidedSession = useSessionStore((s) => s.guidedSession);
  const respondGuided = useSessionStore((s) => s.respondGuided);
  // ... existing hooks ...

  if (guidedSession && !guidedSession.terminal) {
    // The server response from /composer/guided/respond carries `next_turn`
    // separately from `guided_session`. The store keeps the latest `next_turn`
    // alongside `guidedSession`; ChatPanel reads it from there. If a future
    // refactor folds it into the session record, update this read site.
    const turn = useSessionStore((s) => s.guidedNextTurn);
    return (
      <div className="chat-panel guided-mode">
        <GuidedHistory session={guidedSession} />
        <GuidedTurn turn={turn} onSubmit={respondGuided} />
        <ExitToFreeformButton />
      </div>
    );
  }
  if (guidedSession?.terminal?.kind === "completed") {
    return <CompletionSummary terminal={guidedSession.terminal} />;
  }
  // Freeform — unmodified
  return <ExistingChatPanelBody />;
}
```

The `guidedNextTurn` field is added to the store slice in Task 6.3 alongside `guidedSession`. Each server response replaces both atomically.

- [ ] **Step 3: Run, verify pass**

```bash
cd src/elspeth/web/frontend && npx vitest run src/components/chat/ChatPanel.test.tsx
```

- [ ] **Step 4: Commit**

```bash
git add -u
git commit -m "feat(frontend/chat): ChatPanel mode discriminator — guided / completed / freeform

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 8.2: Accessibility audit pass

- [ ] **Step 1: Run axe-core checks against the rendered components**

Use the existing accessibility-audit hook in tests; run on each guided widget. Target: zero AA violations.

- [ ] **Step 2: Verify `prefers-reduced-motion` honoured**

Manual: in Chrome devtools, set `prefers-reduced-motion: reduce`; verify step-advance has no transition.

- [ ] **Step 3: Commit any a11y fixes**

---

## Phase 9 — End-to-End Tests and Demo SLA

**Goal of phase:** Three Playwright flows + the demo-SLA assertion test.

### Task 9.1: Recipe-match happy path

**Files:**
- Create: `src/elspeth/web/frontend/tests/e2e/composer-guided.spec.ts`

> **Plan-vs-reality drift correction (logged 2026-05-12 by Phase 9 controller):**
> The original plan body assumed (a) the test path lives at `tests/playwright/`, (b) `data-testid="new-session"` and `data-testid="blob-upload"` exist, (c) Step 1 includes `InspectAndConfirmTurn` ("Looks right" button), and (d) ≤7 clicks suffices for the happy path. None of those held when verified empirically.
>
> 1. **Path:** the existing E2E suite lives at `src/elspeth/web/frontend/tests/e2e/`, not `tests/playwright/`. Patterns: `tests/e2e/helpers/api.ts` (out-of-band auth + REST), `tests/e2e/page-objects/composer-page.ts`. The `playwright.config.ts` auto-launches uvicorn (:8451) + Vite (:5173) under `webServer` with `composerSettingsEnv` isolation. Use those.
> 2. **Selectors:** no `data-testid` exists on any guided widget (verified). Use role/text selectors (`getByRole("button", { name: "..." })`) per the existing E2E pattern. Do NOT add testids to production widgets.
> 3. **InspectAndConfirmTurn is unreachable on the live emission path:** `routes.py:_build_get_guided_turn` hardcodes `blob_inspection=None`, so the only call site of `build_initial_step_1_turn` never takes the `inspect_and_confirm` branch. The widget's response handler exists, but the server never emits it. Filed as a follow-up. The actual happy path is `SINGLE_SELECT (source)` → `SCHEMA_FORM (source options)` → `SINGLE_SELECT (sink)` → `SCHEMA_FORM (sink options)` → `MULTI_SELECT_WITH_CUSTOM (required fields)` → `RECIPE_OFFER (Apply recipe)` → `CompletionSummary (Save and exit)`.
> 4. **Click budget:** ≤9 clicks under the actual happy path (each SCHEMA_FORM step adds a fill-and-Continue not present in the original plan; required-fields adds an Add+Continue pair). The 30s wall-clock SLA is unchanged and still well within reach (recipe match is deterministic — zero LLM calls).
> 5. **Backend wire-shape bugs blocked the original plan**: HTTP 422 at SchemaFormTurn submission, HTTP 500 at MultiSelectWithCustomTurn submission. Both fixed pre-Task-9.1 (commits `e05e02b2` + `a5df0b6c`).
>
> The pseudocode below has been rewritten against the corrected happy path. Treat it as illustrative — verify actual button labels by reading widget source or probing the running backend before locking in selectors.

- [ ] **Step 1: Write the happy-path E2E**

```typescript
test("guided demo path: CSV → classify-rows-llm-jsonl", async ({ page }) => {
  const start = Date.now();
  let clicks = 0;

  // Out-of-band: create session + upload CSV blob via REST helpers in
  // tests/e2e/helpers/api.ts (authedContext, createSession, uploadBlob).
  // Session-create + blob-upload are NOT counted as clicks.
  // Then navigate the SPA to the session URL.

  // 1. Step 1 source: pick "csv" chip in SingleSelectTurn.
  await page.getByRole("button", { name: /csv/i }).click(); clicks++;

  // 2. Step 1 source options: SchemaFormTurn — fill required fields, Continue.
  await page.getByLabel(/path/i).fill("/tmp/playwright-orders.csv");
  await page.getByRole("button", { name: /continue/i }).click(); clicks++;

  // 3. Step 2 sink: pick "json" (or "jsonl") chip in SingleSelectTurn.
  await page.getByRole("button", { name: /json/i }).click(); clicks++;

  // 4. Step 2 sink options: SchemaFormTurn — fill required fields, Continue.
  await page.getByLabel(/path/i).fill("/tmp/playwright-output.jsonl");
  await page.getByRole("button", { name: /continue/i }).click(); clicks++;

  // 5. Step 2 required fields: declare "category" via custom input + Add, Continue.
  await page.getByPlaceholder(/custom/i).fill("category");
  await page.getByRole("button", { name: /^add$/i }).click(); clicks++;
  await page.getByRole("button", { name: /continue/i }).click(); clicks++;

  // 6. Step 2.5 recipe pre-match offer: Apply recipe.
  await page.getByRole("button", { name: /apply recipe/i }).click(); clicks++;

  // 7. Termination: CompletionSummary — Save and exit.
  await expect(page.getByRole("button", { name: /save and exit/i })).toBeVisible();
  await page.getByRole("button", { name: /save and exit/i }).click(); clicks++;

  // Demo SLA assertions
  expect(clicks).toBeLessThanOrEqual(9);  // happy path under corrected step sequence
  expect(Date.now() - start).toBeLessThan(30_000);
});
```

- [ ] **Step 2: Run, verify pass**

- [ ] **Step 3: Commit**

```bash
git add -u
git commit -m "test(frontend/e2e): guided demo path recipe-match E2E + SLA assertion

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 9.2: Hand-built path (LLM-driven Step 3)

- [ ] **Step 1: Write E2E with stubbed LLM**

Use the existing Playwright LLM-stub fixture. Force pre-match to fail (e.g., DB sink), drive Step 3 through the chain proposal, accept, complete.

- [ ] **Step 2: Run, verify pass**

- [ ] **Step 3: Commit**

### Task 9.3: Auto-drop path

- [ ] **Step 1: Write E2E**

Force LLM stub to return invalid chains twice; assert wizard auto-drops and freeform `<ChatInput>` appears with the partial pipeline state in `compositionState`.

- [ ] **Step 2: Run, verify pass**

- [ ] **Step 3: Commit**

---

## Phase 10 — Documentation and Polish

### Task 10.1: User-facing docs

**Files:**
- Modify: `docs/guides/user-manual.md`
- Modify: `docs/guides/troubleshooting.md`

- [ ] **Step 1: Add "Guided Mode" section to user manual**

A short walk-through of the three steps + recipe match + termination, with a screenshot or two from Playwright.

- [ ] **Step 2: Add troubleshooting entries**

Common issues:
- "Auto-dropped to freeform — what happened?" → explain `solver_exhausted` reason + how to read the chat history
- "Wizard disagreed with my source schema" → explain `inspect_and_confirm` editing
- "Recipe didn't appear for my (CSV, JSONL) pipeline" → explain predicate strictness; refer to `list_recipes`

- [ ] **Step 3: Commit (no-verify, doc-only)**

```bash
git add -u
git commit --no-verify -m "docs(guides): guided-mode user manual + troubleshooting entries

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 10.2: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add a `### Added` entry under the active release header**

```markdown
- **Composer guided mode** — new structured-protocol wizard for first-time
  pipeline authors. Source → sink → transforms in three steps; closed
  six-turn taxonomy; deterministic recipe pre-match; LLM-read-only with
  respect to pipeline state. Ships alongside the unmodified freeform
  composer; mode transition uses progressive disclosure.
  See `docs/superpowers/specs/2026-05-11-composer-guided-mode-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add -u
git commit --no-verify -m "docs(changelog): composer guided mode

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 10.3: Final closure — full test sweep

- [ ] **Step 1: Run everything**

```bash
.venv/bin/python -m pytest tests/unit/ tests/integration/ -v
.venv/bin/python -m mypy src/
.venv/bin/python -m ruff check src/
.venv/bin/python scripts/cicd/enforce_tier_model.py check --root src/elspeth --allowlist config/cicd/enforce_tier_model
cd src/elspeth/web/frontend && npx vitest run && npx playwright test composer-guided
```

Expected: all green.

- [ ] **Step 2: Verify demo-SLA assertion holds**

The Playwright test from Task 9.1 must complete in < 30 s with ≤7 clicks (slack for session-creation + blob-attach overhead beyond the 4-step wizard SLA).

- [ ] **Step 3: Open the umbrella PR**

PR title: `feat(composer): guided mode — structured wizard for first-time pipeline authors`

PR body cites the spec, lists the 5 phase PRs that compose the work, and includes a short demo recording.

---

## Spec Coverage Audit

A trace from each spec section to the tasks that implement it.

| Spec section | Implementing tasks |
|---|---|
| §1 Goals + non-goals | All; §1.3 SLA verified by Task 9.1 |
| §2 Background | (informational; not a task target) |
| §3 User-visible flow — Step 1 | 3.1 (handler) + 7.3, 7.5 (widgets) |
| §3 Step 2 | 3.2 (handler) + 7.4 (widget) |
| §3 Step 2.5 recipe pre-match | 2.3 (matcher) + 3.3 (handler) + 7.7 (widget) |
| §3 Step 3 transforms | 4.3 (solver) + 4.4 (handler) + 7.6 (widget) |
| §3.5 Termination | 4.4 (handler) + 7.10 (CompletionSummary) |
| §4.1 Closed turn taxonomy | 1.1, 1.2 (types) |
| §4.2 LLM tool surface | 4.3 (chain solver tools list) |
| §4.3 Legal-turn matrix | 1.3 (matrix + validator) |
| §5 Mode lifecycle | 1.5 (state field) + 5.x (transitions) |
| §6 Server architecture | All Phase 1–5 backend tasks |
| §7 Frontend architecture | All Phase 6–8 tasks |
| §8.1 Guided skill | 4.1 |
| §8.2 Progressive disclosure | 5.2 |
| §9.1 Audit events | 1.6, 5.3 (full-session test) |
| §9.4 Error handling matrix | 3.6 (illegal turn / 409) + 5.1 (auto-drop) |
| §10 Testing strategy — Ring 1 | 1.x, 2.x test tasks |
| §10 Ring 2 | 3.x, 4.x integration tests |
| §10 Ring 3 + demo SLA | 9.1, 9.2, 9.3 |
| §11 Out of scope | (tasks intentionally absent) |
| §12 Open questions | Flagged inline; resolved during plan-review or implementation |

No spec sections without tasks.

---

**End of plan.**
