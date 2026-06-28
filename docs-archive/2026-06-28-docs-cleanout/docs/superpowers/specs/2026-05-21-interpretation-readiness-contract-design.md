# Interpretation Readiness Contract Design

**Date:** 2026-05-21
**Status:** Implementation-ready draft
**Branch:** `codex/interpretation-readiness-contract`
**Base:** `RC5.2` at `7085501a632ba029ffca7d43bd456fd2b354954d`

---

## 1. Problem

The RC5.2 composer interpretation-review flow currently carries pending human
review state inside `options.prompt_template` as `{{interpretation:<term>}}`.
That sentinel is interpreted in two incompatible ways:

- composer authoring preflight treats it as a pending review token and masks it
  before runtime validation;
- normal `/validate` and LLM plugin configuration treat it as raw Jinja and
  reject it as invalid syntax.

The frontend then feeds `/validate` failures back into the composer, so a valid
pending review handoff becomes a self-sustaining repair loop.

This is a contract bug, not an LLM reachability bug.

## 2. Product Contract

The long-term contract is:

- `authoring_valid`: the draft pipeline is structurally coherent enough for the
  composer and UI to continue.
- `execution_ready`: the pipeline can be launched by the execution service.
- `completion_ready`: the composer may present the current turn as a truthful
  authoring handoff, not necessarily as runnable.

Pending interpretation review means:

- `authoring_valid = true`
- `execution_ready = false`
- `completion_ready = true` only for a review-pending handoff message
- execute buttons and execution endpoints remain blocked until resolution
- composer and frontend must not present the pipeline as runnable/complete
- shareable review may still be saved with a warning, preserving the existing
  `ShareableReviewService` mark-time contract

This preserves the current review-oriented UX while removing the invalid-Jinja
failure mode and the recursive feedback loop.

## 3. Architecture

Interpretation review becomes structured authoring state, not fake prompt text.

```
Composer LLM node options
  prompt_template: real Jinja prompt text only
  interpretation_requirements: structured authoring metadata
      id, user_term, status, prompt_span, draft, event_id, accepted_value, hash
  prompt_template_parts: optional structured prompt-parts source

Validation/readiness
  classify state once in backend
  return typed readiness and typed validation outcomes

Execution
  materialize effective prompt only when all requirements are resolved
  fail unresolved requirements as interpretation_review_pending

Frontend
  render review-pending as first-class state
  do not send validation feedback for interpretation_review_pending
```

### 3.1 Structured Node Metadata

LLM node options gain web-authoring metadata keys:

- `interpretation_requirements`: list of requirement records
- `prompt_template_parts`: optional text/reference prompt-parts list

These keys are web-layer authoring metadata. They must never reach plugin
runtime configuration. The YAML generator strips them before emitting engine
YAML.

`prompt_template` remains a real Jinja template. For a pending review, it may
contain a neutral authoring preview phrase such as `pending interpretation`, but
the source of truth for substituting the accepted meaning is structured state,
not a sentinel substring.

### 3.2 Requirement Shape

```python
class InterpretationRequirement(TypedDict):
    id: str
    user_term: str
    status: Literal["pending", "resolved"]
    draft: str | None
    event_id: str | None
    accepted_value: str | None
    resolved_prompt_template_hash: str | None

class PromptTextPart(TypedDict):
    kind: Literal["text"]
    text: str

class PromptInterpretationRefPart(TypedDict):
    kind: Literal["interpretation_ref"]
    requirement_id: str
```

The requirement `id` is stable within a node. The initial implementation uses a
normalized `user_term` id when unambiguous. If multiple independent requirements
share a term in the same node, the composer must author distinct ids.

### 3.3 Prompt Materialization

Prompt materialization has two modes:

- authoring mode: unresolved refs render as `pending interpretation` so plugin
  config validation can still exercise the real settings/runtime path;
- execution mode: unresolved refs return a typed
  `interpretation_review_pending` readiness blocker and do not build runtime
  YAML.

Resolved refs render the accepted/amended value into the effective prompt. The
`resolved_prompt_template_hash` is computed from the fully materialized resolved
prompt and remains the cross-DB audit anchor that the LLM transform records into
Landscape calls.

### 3.4 Legacy Sentinel Compatibility

Existing session states may still contain `{{interpretation:<term>}}`. The fix
adds a compatibility reader that converts legacy sentinel prompts into
structured requirements in memory for validation and resolution.

New composer instructions and tool validation stop requiring the sentinel.
Newly authored states should use structured metadata instead.

### 3.5 Readiness Result

`ValidationResult` gains an explicit readiness object:

```python
class ValidationReadiness(_StrictResponse):
    authoring_valid: bool
    execution_ready: bool
    completion_ready: bool
    blockers: list[ValidationReadinessBlocker]

class ValidationReadinessBlocker(_StrictResponse):
    code: str
    component_id: str | None
    component_type: str | None
    detail: str
```

`ValidationResult.is_valid` continues to answer "can this validation target be
used for execution?". For pending interpretation review it is `false`, with
`readiness.authoring_valid == true` and a typed
`interpretation_review_pending` blocker. Composer authoring flows use the
readiness object instead of the old `allow_pending_interpretation_placeholders`
split-brain flag.

### 3.6 Frontend Feedback Rule

The frontend validation subscription distinguishes:

- invalid configuration: inject system message and call `sendValidationFeedback`
- pending interpretation review: inject a review-pending status message only
- empty pipeline: suppress message and feedback as today

This stops the recursive compose loop without hiding the pending review from the
operator.

## 4. Touched Areas

Backend:

- `src/elspeth/web/interpretation_state.py` (new structured state helpers)
- `src/elspeth/web/execution/schemas.py`
- `src/elspeth/web/execution/validation.py`
- `src/elspeth/web/execution/service.py`
- `src/elspeth/web/composer/tools.py`
- `src/elspeth/web/composer/service.py`
- `src/elspeth/web/composer/yaml_generator.py`
- `src/elspeth/web/composer/skills/pipeline_composer.md`
- `src/elspeth/web/sessions/service.py`
- `src/elspeth/web/audit_readiness/service.py`

Frontend:

- `src/elspeth/web/frontend/src/types/index.ts`
- `src/elspeth/web/frontend/src/stores/subscriptions.ts`
- `src/elspeth/web/frontend/src/stores/subscriptions.test.ts`
- `src/elspeth/web/frontend/src/api/auditReadiness.ts`
- validation/readiness renderers if strict type guards require updates

## 5. Test Strategy

Tests must use production paths, not handcrafted shortcuts:

- schema tests for `ValidationReadiness`
- unit tests for structured interpretation state parsing and materialization
- execution validation tests proving pending review returns
  `interpretation_review_pending`, not invalid Jinja
- execution service tests proving unresolved review blocks before run creation
- session service tests proving resolution updates structured state and hashes
  the materialized prompt
- composer tool tests proving `request_interpretation_review` binds structured
  requirements and legacy sentinel states still work during migration
- composer finalization tests proving pending review produces a truthful
  handoff and no repair loop
- audit-readiness tests proving raw `validation_result` carries readiness and
  warning rows remain non-blocking for shareable review
- frontend subscription tests proving pending review does not call
  `sendValidationFeedback`

## 6. Non-Goals

- No database schema migration is required for the first implementation because
  structured interpretation metadata lives in existing `composition_states.nodes`
  JSON.
- No live Azure staging redeploy is part of this implementation unless the user
  separately requests it.
- No broad composer UX redesign beyond rendering pending review as a typed
  readiness state.

## 7. Risks And Controls

- `ValidationResult` is strict and widely constructed in tests. The plan uses a
  small helper factory in backend tests and explicit construction updates in
  production code to avoid silent default drift.
- Authoring metadata in node options would break plugin config if leaked. The
  YAML generator strips these keys and tests assert they do not appear in YAML.
- Legacy sentinel support can perpetuate the old design if not bounded. New
  skill guidance and tool validation prefer structured metadata; legacy
  conversion is read/migration support only.
- Shareable review currently permits warning rows. The implementation preserves
  that behavior while keeping execution readiness false.
