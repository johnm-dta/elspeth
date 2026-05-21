# Interpretation Readiness Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace interpretation-review sentinel strings with a backend-owned structured readiness contract so pending human review stops producing validation feedback loops and never reaches runtime as invalid Jinja.

**Architecture:** Add structured interpretation metadata helpers, materialize prompts explicitly for authoring versus execution, extend `ValidationResult` with readiness, and align composer, execution, audit-readiness, and frontend validation feedback on the same typed pending-review outcome. Existing shareable-review warning behavior is preserved; execution remains strict.

**Tech Stack:** Python 3.13, Pydantic v2, FastAPI, SQLAlchemy Core, React 18, TypeScript, pytest, Vitest, uv.

**Spec:** `docs/superpowers/specs/2026-05-21-interpretation-readiness-contract-design.md`

---

## Implementation Invariants

- `prompt_template` is runtime prompt text only.
- Structured interpretation metadata lives in L3 web authoring state and is
  stripped before engine/plugin runtime configuration.
- Pending review is not execution-ready and must not create a run.
- Pending review is not sent back into the composer as validation feedback.
- `resolved_prompt_template_hash` is computed from the materialized resolved
  prompt.
- Tests must exercise production paths: `validate_pipeline`, `ExecutionService`,
  composer tool handlers, session service writers, and frontend store
  subscriptions.

## File Structure

### Backend New Files

- `src/elspeth/web/interpretation_state.py` — structured interpretation metadata
  constants, typed dictionaries, legacy sentinel conversion, prompt
  materialization, readiness-site detection.
- `tests/unit/web/test_interpretation_state.py` — focused helper tests.

### Backend Modified Files

- `src/elspeth/web/execution/schemas.py` — add `ValidationReadiness` and
  `ValidationReadinessBlocker`; add required `readiness` to `ValidationResult`.
- `src/elspeth/web/execution/validation.py` — replace masking split-brain with
  authoring/execution materialization and typed pending-review blocker.
- `src/elspeth/web/execution/service.py` — use execution materialization before
  execute; keep unresolved review as pre-run error.
- `src/elspeth/web/composer/yaml_generator.py` — strip authoring metadata and
  accept already-materialized runtime prompt options.
- `src/elspeth/web/composer/tools.py` — bind `request_interpretation_review` to
  structured requirements; retain legacy sentinel compatibility.
- `src/elspeth/web/composer/service.py` — consume readiness in runtime preflight
  and finalization; stop repair-loop logic from treating pending review as a
  configuration error.
- `src/elspeth/web/composer/skills/pipeline_composer.md` — teach structured
  interpretation metadata instead of sentinel placeholders.
- `src/elspeth/web/sessions/service.py` — resolve requirements by structured id,
  materialize prompt, update hash, preserve legacy sentinel resolution.
- `src/elspeth/web/audit_readiness/service.py` — preserve raw readiness in
  snapshot and keep warning row behavior.

### Frontend Modified Files

- `src/elspeth/web/frontend/src/types/index.ts` — add readiness types.
- `src/elspeth/web/frontend/src/api/auditReadiness.ts` — validate readiness
  fields in raw `validation_result`.
- `src/elspeth/web/frontend/src/stores/subscriptions.ts` — suppress
  `sendValidationFeedback` for typed pending review while showing a status
  message.
- `src/elspeth/web/frontend/src/stores/subscriptions.test.ts` — add regression
  coverage for the loop.

### Test Files Modified Or Added

- `tests/unit/web/execution/test_schemas.py`
- `tests/unit/web/execution/test_validation.py`
- `tests/unit/web/execution/test_service.py`
- `tests/unit/web/composer/test_request_interpretation_review_tool.py`
- `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`
- `tests/unit/web/composer/test_service.py`
- `tests/unit/web/sessions/test_interpretation_events_service.py`
- `tests/unit/web/audit_readiness/test_service.py`
- `tests/unit/web/shareable_reviews/test_service.py`

## Phase 0: Worktree Setup And Baseline

- [ ] **Step 0.1: Create or verify worktree**

Run:

```bash
git status --short --branch
git rev-parse HEAD
```

Expected:

```text
## codex/interpretation-readiness-contract
7085501a632ba029ffca7d43bd456fd2b354954d
```

- [ ] **Step 0.2: Create a proper worktree-local Python environment**

Run from the worktree root:

```bash
uv sync --frozen --extra all --extra webui
```

Expected: command exits 0 and creates `.venv/`.

If `uv` starts a partial environment and fails, stop and report instead of
symlinking around it.

- [ ] **Step 0.3: Install frontend dependencies**

Run:

```bash
cd src/elspeth/web/frontend
npm install
```

Expected: command exits 0 and creates `node_modules/`.

- [ ] **Step 0.4: Run targeted baseline**

Run:

```bash
uv run pytest \
  tests/unit/web/execution/test_validation.py::TestValidatePipelinePendingInterpretationPlaceholders \
  tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py::test_composer_runtime_preflight_allows_pending_interpretation_placeholders \
  tests/unit/web/execution/test_service.py::TestExecuteUnresolvedInterpretationPlaceholderGate::test_execute_rejects_unresolved_placeholder_before_creating_run
```

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/stores/subscriptions.test.ts
```

Expected: current baseline passes before behavior changes.

## Phase 1: Readiness Schema

- [ ] **Step 1.1: Write failing schema tests**

Add tests to `tests/unit/web/execution/test_schemas.py`:

```python
def test_validation_result_requires_readiness():
    with pytest.raises(ValidationError):
        ValidationResult(is_valid=True, checks=[], errors=[], semantic_contracts=[])


def test_validation_readiness_accepts_pending_interpretation_blocker():
    result = ValidationResult(
        is_valid=False,
        checks=[],
        errors=[
            ValidationError(
                component_id="rate_coolness",
                component_type="transform",
                message="Interpretation review is pending for 'coolness'.",
                suggestion="Resolve the pending interpretation review before running.",
                error_code="interpretation_review_pending",
            )
        ],
        semantic_contracts=[],
        readiness=ValidationReadiness(
            authoring_valid=True,
            execution_ready=False,
            completion_ready=True,
            blockers=[
                ValidationReadinessBlocker(
                    code="interpretation_review_pending",
                    component_id="rate_coolness",
                    component_type="transform",
                    detail="coolness",
                )
            ],
        ),
    )
    assert result.readiness.authoring_valid is True
    assert result.readiness.execution_ready is False
```

Run:

```bash
uv run pytest tests/unit/web/execution/test_schemas.py::test_validation_result_requires_readiness tests/unit/web/execution/test_schemas.py::test_validation_readiness_accepts_pending_interpretation_blocker -v
```

Expected: fails because readiness models do not exist or are not required.

- [ ] **Step 1.2: Implement readiness models**

Modify `src/elspeth/web/execution/schemas.py`:

```python
class ValidationReadinessBlocker(_StrictResponse):
    code: str
    component_id: str | None
    component_type: str | None
    detail: str


class ValidationReadiness(_StrictResponse):
    authoring_valid: bool
    execution_ready: bool
    completion_ready: bool
    blockers: list[ValidationReadinessBlocker]


class ValidationResult(_StrictResponse):
    is_valid: bool
    checks: list[ValidationCheck]
    errors: list[ValidationError]
    readiness: ValidationReadiness
    semantic_contracts: list[SemanticEdgeContractResponse] = []
```

Add small helper constructors in `src/elspeth/web/execution/validation.py`:

```python
def _ready() -> ValidationReadiness:
    return ValidationReadiness(authoring_valid=True, execution_ready=True, completion_ready=True, blockers=[])


def _not_authoring_ready(code: str, detail: str) -> ValidationReadiness:
    return ValidationReadiness(
        authoring_valid=False,
        execution_ready=False,
        completion_ready=False,
        blockers=[ValidationReadinessBlocker(code=code, component_id=None, component_type=None, detail=detail)],
    )
```

Update every production `ValidationResult(...)` construction to pass explicit
readiness. Update tests with helper factories rather than defaults.

Run:

```bash
uv run pytest tests/unit/web/execution/test_schemas.py -v
```

Expected: schema tests pass.

## Phase 2: Structured Interpretation State Helpers

- [ ] **Step 2.1: Write failing helper tests**

Create `tests/unit/web/test_interpretation_state.py`:

```python
from dataclasses import replace

from elspeth.core.hashing import stable_hash
from elspeth.web.composer.state import CompositionState, NodeSpec
from elspeth.web.interpretation_state import (
    AUTHORING_METADATA_OPTION_KEYS,
    InterpretationReviewPending,
    interpretation_sites,
    materialize_state_for_authoring,
    materialize_state_for_execution,
    strip_authoring_options,
)


def _state_with_llm(options):
    return CompositionState(
        source=None,
        nodes=(
            NodeSpec(
                id="rate_coolness",
                node_type="transform",
                plugin="llm",
                input="source",
                on_success="output",
                on_error="stop",
                options=options,
            ),
        ),
        edges=(),
        outputs=(),
        version=1,
    )


def test_legacy_placeholder_materializes_for_authoring_without_mutating_source():
    state = _state_with_llm({"prompt_template": "Rate {{interpretation:coolness}}: {{ row.text }}"})
    authoring = materialize_state_for_authoring(state)
    assert authoring.nodes[0].options["prompt_template"] == "Rate pending interpretation: {{ row.text }}"
    assert state.nodes[0].options["prompt_template"] == "Rate {{interpretation:coolness}}: {{ row.text }}"


def test_pending_structured_requirement_blocks_execution_with_typed_site():
    state = _state_with_llm(
        {
            "prompt_template": "Rate pending interpretation: {{ row.text }}",
            "prompt_template_parts": [
                {"kind": "text", "text": "Rate "},
                {"kind": "interpretation_ref", "requirement_id": "coolness"},
                {"kind": "text", "text": ": {{ row.text }}"},
            ],
            "interpretation_requirements": [
                {
                    "id": "coolness",
                    "user_term": "coolness",
                    "status": "pending",
                    "draft": "well-designed and useful",
                    "event_id": "event-1",
                    "accepted_value": None,
                    "resolved_prompt_template_hash": None,
                }
            ],
        }
    )
    result = materialize_state_for_execution(state)
    assert isinstance(result, InterpretationReviewPending)
    assert result.sites == (("rate_coolness", "coolness"),)


def test_resolved_requirement_materializes_prompt_and_hash():
    state = _state_with_llm(
        {
            "prompt_template": "Rate pending interpretation: {{ row.text }}",
            "prompt_template_parts": [
                {"kind": "text", "text": "Rate "},
                {"kind": "interpretation_ref", "requirement_id": "coolness"},
                {"kind": "text", "text": ": {{ row.text }}"},
            ],
            "interpretation_requirements": [
                {
                    "id": "coolness",
                    "user_term": "coolness",
                    "status": "resolved",
                    "draft": "well-designed and useful",
                    "event_id": "event-1",
                    "accepted_value": "well-designed and useful",
                    "resolved_prompt_template_hash": None,
                }
            ],
        }
    )
    materialized = materialize_state_for_execution(state)
    prompt = materialized.nodes[0].options["prompt_template"]
    assert prompt == "Rate well-designed and useful: {{ row.text }}"
    assert materialized.nodes[0].options["resolved_prompt_template_hash"] == stable_hash(prompt)


def test_strip_authoring_options_removes_metadata_keys():
    options = {
        "prompt_template": "Rate {{ row.text }}",
        "prompt_template_parts": [],
        "interpretation_requirements": [],
        "resolved_prompt_template_hash": "a" * 64,
    }
    stripped = strip_authoring_options(options)
    assert "prompt_template_parts" not in stripped
    assert "interpretation_requirements" not in stripped
    assert stripped["resolved_prompt_template_hash"] == "a" * 64
```

Run:

```bash
uv run pytest tests/unit/web/test_interpretation_state.py -v
```

Expected: fails because `elspeth.web.interpretation_state` does not exist.

- [ ] **Step 2.2: Implement helper module**

Create `src/elspeth/web/interpretation_state.py` with:

- constants:
  - `INTERPRETATION_REQUIREMENTS_KEY`
  - `PROMPT_TEMPLATE_PARTS_KEY`
  - `AUTHORING_METADATA_OPTION_KEYS`
  - `INTERPRETATION_REVIEW_PENDING_CODE`
- frozen dataclass `InterpretationReviewPending`
- parsing helpers that assert Tier-1 state shape rather than fabricating
  defaults
- `materialize_state_for_authoring(state) -> CompositionState`
- `materialize_state_for_execution(state) -> CompositionState | InterpretationReviewPending`
- `interpretation_sites(nodes) -> tuple[tuple[str, str], ...]`
- `strip_authoring_options(options) -> dict[str, Any]`

Implementation details:

- Use the existing `INTERPRETATION_PLACEHOLDER_RE` pattern or move it into this
  module to avoid duplicate regexes.
- For legacy sentinel prompts, authoring mode masks to `pending interpretation`
  and execution mode returns `InterpretationReviewPending`.
- For structured prompt parts, concatenate text parts and resolved accepted
  values. If any requirement is pending, return `InterpretationReviewPending`.
- Compute `resolved_prompt_template_hash` with `stable_hash(prompt)`.

Run:

```bash
uv run pytest tests/unit/web/test_interpretation_state.py -v
```

Expected: pass.

## Phase 3: Validation Contract

- [ ] **Step 3.1: Write failing validation tests**

Modify `tests/unit/web/execution/test_validation.py`:

```python
def test_validate_pending_structured_interpretation_returns_typed_readiness(settings):
    state = _state_with_llm_transform(
        prompt_template="Rate pending interpretation: {{ row.text }}",
        extra_options={
            "prompt_template_parts": [
                {"kind": "text", "text": "Rate "},
                {"kind": "interpretation_ref", "requirement_id": "coolness"},
                {"kind": "text", "text": ": {{ row.text }}"},
            ],
            "interpretation_requirements": [
                {
                    "id": "coolness",
                    "user_term": "coolness",
                    "status": "pending",
                    "draft": "well-designed and useful",
                    "event_id": "event-1",
                    "accepted_value": None,
                    "resolved_prompt_template_hash": None,
                }
            ],
        },
    )

    result = validate_pipeline(state, settings, yaml_generator)

    assert result.is_valid is False
    assert result.errors[0].error_code == "interpretation_review_pending"
    assert "Invalid Jinja2 template" not in result.errors[0].message
    assert result.readiness.authoring_valid is True
    assert result.readiness.execution_ready is False
```

Run:

```bash
uv run pytest tests/unit/web/execution/test_validation.py::test_validate_pending_structured_interpretation_returns_typed_readiness -v
```

Expected: fail because validation does not inspect structured requirements.

- [ ] **Step 3.2: Implement validation materialization**

Modify `src/elspeth/web/execution/validation.py`:

- call `materialize_state_for_execution(state)` near the start, after
  empty-pipeline handling and before YAML generation;
- if it returns `InterpretationReviewPending`, return `ValidationResult` with:
  - `is_valid=False`
  - `error_code="interpretation_review_pending"`
  - `readiness.authoring_valid=True`
  - `readiness.execution_ready=False`
  - `readiness.completion_ready=True`
- remove dependence on `allow_pending_interpretation_placeholders` in normal
  composer flow after composer service is updated;
- keep legacy flag only as a temporary compatibility shim until all call sites
  use readiness.

Run:

```bash
uv run pytest tests/unit/web/execution/test_validation.py -v
```

Expected: validation tests pass.

## Phase 4: YAML Generator Runtime Hygiene

- [ ] **Step 4.1: Write failing YAML strip test**

Add to `tests/unit/web/test_interpretation_state.py` or existing YAML tests:

```python
def test_yaml_generator_strips_interpretation_authoring_metadata():
    state = _state_with_llm(
        {
            "prompt_template": "Rate resolved meaning: {{ row.text }}",
            "prompt_template_parts": [{"kind": "text", "text": "ignored"}],
            "interpretation_requirements": [],
            "resolved_prompt_template_hash": "a" * 64,
        }
    )
    doc = generate_pipeline_dict(state)
    options = doc["transforms"][0]["options"]
    assert "prompt_template_parts" not in options
    assert "interpretation_requirements" not in options
    assert options["prompt_template"] == "Rate resolved meaning: {{ row.text }}"
```

Run:

```bash
uv run pytest tests/unit/web/test_interpretation_state.py::test_yaml_generator_strips_interpretation_authoring_metadata -v
```

Expected: fail because transform options are not stripped today.

- [ ] **Step 4.2: Strip authoring metadata in generator**

Modify `src/elspeth/web/composer/yaml_generator.py`:

```python
_WEB_ONLY_OPTION_KEYS = frozenset(
    {
        "blob_ref",
        "prompt_template_parts",
        "interpretation_requirements",
    }
)
```

Apply `_strip_web_metadata(dict(t["options"]))` for transform options as well
as source options.

Run:

```bash
uv run pytest tests/unit/web/test_interpretation_state.py::test_yaml_generator_strips_interpretation_authoring_metadata -v
```

Expected: pass.

## Phase 5: Execution Service Gate

- [ ] **Step 5.1: Write failing execution test**

Modify `tests/unit/web/execution/test_service.py` to add a structured pending
state beside the legacy sentinel test:

```python
async def test_execute_rejects_structured_pending_interpretation_before_creating_run(...):
    state_record = await insert_state_with_structured_pending_interpretation(...)
    with pytest.raises(UnresolvedInterpretationPlaceholderError) as excinfo:
        await service.execute(session_id, user_id=user_id)
    assert "coolness" in str(excinfo.value)
    assert no_run_rows_exist()
```

Run the test and expect failure because execution only scans sentinel strings.

- [ ] **Step 5.2: Implement structured execution gate**

Modify `src/elspeth/web/execution/service.py`:

- replace direct `_detect_unresolved_interpretation_placeholders_typed` usage
  with `materialize_state_for_execution`;
- if pending, raise the existing `UnresolvedInterpretationPlaceholderError`
  with sites from structured state;
- if materialized, pass the materialized state to validation/YAML execution.

Run:

```bash
uv run pytest tests/unit/web/execution/test_service.py::TestExecuteUnresolvedInterpretationPlaceholderGate -v
```

Expected: pass.

## Phase 6: Composer Tool Binding

- [ ] **Step 6.1: Write failing composer tool tests**

Modify `tests/unit/web/composer/test_request_interpretation_review_tool.py`:

- add a node with `interpretation_requirements` and no sentinel;
- assert `_handle_request_interpretation_review` succeeds for matching
  `affected_node_id` and `user_term`;
- assert wrong `user_term` fails with a structured tool argument error;
- keep legacy sentinel tests passing during migration.

Run:

```bash
uv run pytest tests/unit/web/composer/test_request_interpretation_review_tool.py -k "structured or interpretation_review" -v
```

Expected: structured test fails because `_assert_affected_llm_node` still
requires `{{interpretation:<term>}}`.

- [ ] **Step 6.2: Implement structured binding**

Modify `src/elspeth/web/composer/tools.py`:

- replace `_assert_affected_llm_node` internals with:
  1. verify production node shape: `node_type == "transform"` and
     `plugin == "llm"`;
  2. if structured requirements exist, require exactly one pending requirement
     matching `user_term`;
  3. else fall back to legacy sentinel matching;
  4. return a normalized requirement id for event creation if needed.
- update `_detect_unresolved_interpretation_placeholders_typed` to use
  `interpretation_sites`.

Run:

```bash
uv run pytest tests/unit/web/composer/test_request_interpretation_review_tool.py -v
```

Expected: pass.

## Phase 7: Session Resolution

- [ ] **Step 7.1: Write failing session resolution tests**

Modify `tests/unit/web/sessions/test_interpretation_events_service.py`:

- add a current state with structured pending requirement and prompt parts;
- create pending event;
- resolve event;
- assert:
  - the new composition state has `prompt_template` materialized with accepted
    value;
  - requirement status is `resolved`;
  - `accepted_value` is stored in structured metadata;
  - `resolved_prompt_template_hash == stable_hash(materialized_prompt)`;
  - generated YAML contains no authoring metadata.

Run:

```bash
uv run pytest tests/unit/web/sessions/test_interpretation_events_service.py -k "structured" -v
```

Expected: fail because resolution only patches sentinel strings.

- [ ] **Step 7.2: Implement structured resolution**

Modify `src/elspeth/web/sessions/service.py`:

- replace `_patch_llm_transform_prompt` with a wrapper that:
  - resolves structured requirement when present;
  - falls back to legacy sentinel patch when no structured requirement exists;
  - uses materialization helper to compute prompt/hash.
- preserve the closed structural directive guard for legacy sentinel fallback.
- keep provenance `interpretation_resolve`.

Run:

```bash
uv run pytest tests/unit/web/sessions/test_interpretation_events_service.py -v
```

Expected: pass.

## Phase 8: Composer Finalization And Preflight

- [ ] **Step 8.1: Write failing composer finalization tests**

Modify `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`
or `tests/unit/web/composer/test_service.py`:

- structured pending review should produce a `runtime_preflight` with
  `readiness.authoring_valid=True` and `execution_ready=False`;
- final message should preserve the assistant handoff but append/no-op with
  review-pending framing, not "marked complete";
- no repair message should be injected when a pending event exists and matches
  the structured requirement.

Run:

```bash
uv run pytest tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py -k "structured or pending_interpretation" -v
```

Expected: fail before service consumes readiness.

- [ ] **Step 8.2: Implement readiness-aware composer finalization**

Modify `src/elspeth/web/composer/service.py`:

- change `_runtime_preflight` to call validation without masking sentinel
  placeholders;
- update missing pending review detection to use structured sites and pending
  events;
- update `_finalize_no_tool_response` to branch on
  `runtime_result.readiness.blockers`;
- pending review should return a truthful handoff message and avoid
  `sendValidationFeedback` loops, but should not call the pipeline runnable.

Run:

```bash
uv run pytest tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py tests/unit/web/composer/test_service.py -k "interpretation or preflight" -v
```

Expected: pass.

## Phase 9: Frontend Feedback Contract

- [ ] **Step 9.1: Write failing frontend test**

Modify `src/elspeth/web/frontend/src/stores/subscriptions.test.ts`:

```typescript
it("does not send validation feedback for pending interpretation review", async () => {
  initStoreSubscriptions();
  const result: ValidationResult = {
    is_valid: false,
    checks: [],
    errors: [
      {
        component_id: "rate_coolness",
        component_type: "transform",
        message: "Interpretation review is pending for 'coolness'.",
        suggestion: "Resolve the pending interpretation review before running.",
        error_code: "interpretation_review_pending",
      },
    ],
    semantic_contracts: [],
    readiness: {
      authoring_valid: true,
      execution_ready: false,
      completion_ready: true,
      blockers: [
        {
          code: "interpretation_review_pending",
          component_id: "rate_coolness",
          component_type: "transform",
          detail: "coolness",
        },
      ],
    },
  };

  useExecutionStore.getState().setValidationResult(result);

  expect(useSessionStore.getState().injectSystemMessage).toHaveBeenCalled();
  expect(useSessionStore.getState().sendValidationFeedback).not.toHaveBeenCalled();
});
```

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/stores/subscriptions.test.ts
```

Expected: fail because the subscription still sends feedback for any invalid
result with errors.

- [ ] **Step 9.2: Implement frontend guard and types**

Modify:

- `src/elspeth/web/frontend/src/types/index.ts`
- `src/elspeth/web/frontend/src/api/auditReadiness.ts`
- `src/elspeth/web/frontend/src/stores/subscriptions.ts`

Add:

```typescript
function isInterpretationReviewPendingResult(result: ValidationResult): boolean {
  return (
    !result.is_valid &&
    result.readiness.blockers.length > 0 &&
    result.readiness.blockers.every((blocker) => blocker.code === "interpretation_review_pending")
  );
}
```

Use this guard before the generic invalid-feedback branch. Inject a short
review-pending system message and do not call `sendValidationFeedback`.

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/stores/subscriptions.test.ts
```

Expected: pass.

## Phase 10: Skill Guidance And Legacy Drift Tests

- [ ] **Step 10.1: Update composer skill drift tests first**

Modify `tests/unit/web/composer/test_skill_drift.py` so the expected guidance:

- no longer requires `{{interpretation:<term>}}` as the primary instruction;
- requires structured interpretation metadata / prompt parts language;
- still requires `request_interpretation_review`.

Run:

```bash
uv run pytest tests/unit/web/composer/test_skill_drift.py -v
```

Expected: fail until skill markdown is updated.

- [ ] **Step 10.2: Update skill markdown**

Modify `src/elspeth/web/composer/skills/pipeline_composer.md`:

- replace sentinel-first staging guidance with structured
  `interpretation_requirements` and `prompt_template_parts`;
- retain legacy warning only in migration/internal notes if needed;
- keep opt-out behavior unchanged.

Run:

```bash
uv run pytest tests/unit/web/composer/test_skill_drift.py -v
```

Expected: pass.

## Phase 11: Audit Readiness And Shareable Review

- [ ] **Step 11.1: Write/update backend tests**

Modify:

- `tests/unit/web/audit_readiness/test_service.py`
- `tests/unit/web/shareable_reviews/test_service.py`

Assert:

- audit-readiness snapshot preserves `validation_result.readiness`;
- `llm_interpretations` warning row remains warning for pending review;
- `mark_ready_for_review` still permits warning rows;
- execution readiness false does not get misreported as malformed Jinja.

Run:

```bash
uv run pytest tests/unit/web/audit_readiness/test_service.py tests/unit/web/shareable_reviews/test_service.py -k "interpretation or validation or warning" -v
```

Expected: fail until constructors/types are updated.

- [ ] **Step 11.2: Update services/tests**

Modify services only if needed to pass the explicit contract. The preferred
change is construction/type propagation only; do not make warning rows block
shareable review.

Run:

```bash
uv run pytest tests/unit/web/audit_readiness/test_service.py tests/unit/web/shareable_reviews/test_service.py -v
```

Expected: pass.

## Phase 12: Focused Regression Sweep

- [ ] **Step 12.1: Run backend focused suite**

Run:

```bash
uv run pytest \
  tests/unit/web/test_interpretation_state.py \
  tests/unit/web/execution/test_schemas.py \
  tests/unit/web/execution/test_validation.py \
  tests/unit/web/execution/test_service.py::TestExecuteUnresolvedInterpretationPlaceholderGate \
  tests/unit/web/composer/test_request_interpretation_review_tool.py \
  tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py \
  tests/unit/web/sessions/test_interpretation_events_service.py \
  tests/unit/web/audit_readiness/test_service.py \
  tests/unit/web/shareable_reviews/test_service.py
```

Expected: pass.

- [ ] **Step 12.2: Run frontend focused suite**

Run:

```bash
cd src/elspeth/web/frontend
npm run test -- src/stores/subscriptions.test.ts src/components/audit/AuditReadinessPanel.test.tsx src/components/sidebar/ExecuteButton.test.tsx
```

Expected: pass.

## Phase 13: Broader Verification

- [ ] **Step 13.1: Run lint/type checks appropriate to touched files**

Run:

```bash
uv run ruff check src/elspeth/web/execution src/elspeth/web/composer src/elspeth/web/sessions src/elspeth/web/audit_readiness tests/unit/web
```

Run:

```bash
cd src/elspeth/web/frontend
npm run typecheck
```

Expected: pass.

- [ ] **Step 13.2: Run frontend build**

Run:

```bash
cd src/elspeth/web/frontend
npm run build
```

Expected: pass. Existing Vite chunk-size warnings are acceptable if unchanged.

## Phase 14: Completion Audit

- [ ] **Step 14.1: Search for old split-brain paths**

Run:

```bash
rg -n "allow_pending_interpretation_placeholders|mask_pending_interpretation|{{interpretation:|Invalid Jinja2 template" src/elspeth tests
```

Expected:

- no active authoring path depends on sentinel masking;
- sentinel references remain only in legacy compatibility tests/comments/errors;
- no pending-review validation path returns an invalid-Jinja message.

- [ ] **Step 14.2: Inspect git diff**

Run:

```bash
git diff --stat
git diff --check
git status --short
```

Expected:

- no whitespace errors;
- only intended files changed.

- [ ] **Step 14.3: Summarize readiness evidence**

Document in final response:

- backend typed readiness added and verified;
- structured interpretation state implemented;
- `/validate` and composer preflight aligned;
- frontend loop stopped;
- execution remains strict;
- shareable review warning contract preserved;
- tests run and any limitations.
