# Tutorial Assumption Surfacing And Graduation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tutorial and normal composer sessions surface all LLM-authored assumptions - vague terms, invented source data, and LLM prompt templates - before execution, then add a final graduation turn that records tutorial completion.

**Architecture:** Add `InterpretationKind` as the discriminator that travels through the L0 event contract, session DB, composer session-aware tool, structured authoring metadata, runtime preflight, and frontend review card. Keep `request_interpretation_review` on the current session-aware async path; do not force it into `ToolDeclaration` until `elspeth-f5da936747` lands. Treat the spec's "no database schema migration" line as superseded: this requires a session schema epoch bump because `interpretation_events` must persist `kind` and surface-specific opt-out rows.

**Tech Stack:** Python 3.13, SQLAlchemy Core, Pydantic v2, FastAPI, React, TypeScript, Zustand, Vitest, Playwright, pytest, ruff, mypy, elspeth-lints.

---

## Inputs And Review Corrections

Spec: `docs/superpowers/specs/2026-05-22-tutorial-assumption-surfacing-and-graduation-design.md`

Recent-diff corrections applied to this plan:

- The last 36 hours moved composer tool ownership to `ToolDeclaration`, but `request_interpretation_review` remains a documented session-aware async carve-out. Update its inline definition in `src/elspeth/web/composer/tools/_dispatch.py`, its async handler in `src/elspeth/web/composer/tools/sessions.py`, and its redaction model in `src/elspeth/web/composer/redaction.py`; do not add a normal declaration.
- `src/elspeth/web/interpretation_state.py` already has structured `interpretation_requirements` and `prompt_template_parts`. Extend that structure with `kind`, source metadata, and review-site objects; do not re-center the design on legacy `{{interpretation:<term>}}` placeholders.
- `create_blob` and `set_pipeline.source.inline_blob` currently mark LLM-provided inline content as `CreationModality.VERBATIM`. Class 2 cannot be enforced until composer-authored source provenance is mechanical. Add the ToolContext provenance fields and blob/source metadata first.
- The current `interpretation_events` constraints reject `AUTO_INTERPRETED_OPT_OUT` rows that carry `kind`, `accepted_value`, or LLM provenance. A schema epoch bump is mandatory.
- Frontend tutorial CSS now lives in `src/elspeth/web/frontend/src/components/tutorial/tutorial.css`; do not edit deleted `App.css`.

## Implementation Invariants

- `InterpretationKind` is a closed L0 enum in `src/elspeth/contracts/composer_interpretation.py`.
- `hash_domain_version` uses `"v2"` for resolved surface rows after this change. `INTERPRETATION_HASH_DOMAIN_V1` is retired from write paths.
- Existing session-level opt-out marker rows remain valid: `interpretation_source=auto_interpreted_opt_out`, `kind IS NULL`, no surface fields, no hash.
- New surface-specific opt-out rows are born resolved: `interpretation_source=auto_interpreted_opt_out`, `kind IS NOT NULL`, surface fields populated, `accepted_value` populated, and V2 hash populated.
- `request_interpretation_review` validates Tier-3 arguments at the tool boundary. `llm_prompt_template` drafts are allowed to contain Jinja; vague-term and invented-source drafts still use the existing accepted-value validator.
- Runtime/YAML output never contains web-only authoring metadata. Add new metadata keys to `AUTHORING_METADATA_OPTION_KEYS`; the YAML generator already strips that set.
- No row-level decision goes to logger/structlog. Interpretation decisions remain audit-primary rows; telemetry additions are operational only.
- Tests must use production paths: compose-loop tests go through `_run_one_turn_for_test`; handler-only tests are used only for handler-local argument validation.

## File Structure

Contract and session DB:

- Modify `src/elspeth/contracts/composer_interpretation.py` - add `InterpretationKind`, add `kind` to `InterpretationEventRecord`, replace active hash domain with V2.
- Modify `src/elspeth/web/sessions/models.py` - bump `SESSION_SCHEMA_EPOCH`, add `interpretation_events.kind`, update CHECK constraints for session opt-out marker rows and surface opt-out rows.
- Modify `src/elspeth/web/sessions/service.py` - hydrate/write `kind`, branch create/resolve/opt-out logic per kind, compute V2 hashes.
- Modify `src/elspeth/web/sessions/schemas.py`, `src/elspeth/web/sessions/routes/_helpers.py`, `src/elspeth/web/sessions/protocol.py` - expose `kind` on wire and service protocols.

Composer and authoring state:

- Modify `src/elspeth/web/interpretation_state.py` - extend structured requirements with `kind`, add source authoring metadata, return kind-aware review sites, materialize per kind.
- Modify `src/elspeth/web/composer/tools/_common.py` and `_dispatch.py` - thread composer provenance into `ToolContext`.
- Modify `src/elspeth/web/composer/tools/blobs.py` and `sessions.py` - classify LLM-generated inline blobs, stamp source metadata, update `request_interpretation_review`.
- Modify `src/elspeth/web/composer/redaction.py` and `tests/unit/web/composer/redaction_policy_snapshot.json` - add `kind` to redaction-bearing model and regenerate snapshot.
- Modify `src/elspeth/web/composer/skills/pipeline_composer.md` - replace chat-narration-only assumption guidance.
- Modify `src/elspeth/web/composer/yaml_generator.py` only if new source metadata is not already covered by `AUTHORING_METADATA_OPTION_KEYS`.

Execution and validation:

- Modify `src/elspeth/web/execution/service.py`, `src/elspeth/web/execution/validation.py`, and related schemas/tests - report kind-aware blockers and enforce missing Class 2/Class 3 defense-in-depth.

Frontend:

- Modify `src/elspeth/web/frontend/src/types/interpretation.ts` and API/client tests - add `InterpretationKind` and `kind`.
- Modify `src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.tsx` and `src/elspeth/web/frontend/src/hooks/useInterpretationResolver.ts` - kind-aware copy, opt-out/amend visibility, prompt-template scroll gate.
- Modify `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx`, `TutorialTurn2bShowBuilt.tsx`, `TutorialTurn6ModeChoice.tsx`, `HelloWorldTutorial.tsx`, `tutorialMachine.ts`, `copy.ts`, and `tutorial.css`.
- Create `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.tsx`.
- Modify `src/elspeth/web/frontend/src/stores/preferencesStore.ts` - split mode save from tutorial graduation completion.

## Task 0: Create The Worktree And Confirm Baseline

**Files:**

- No code files changed in this task.

- [ ] **Step 1: Create an isolated worktree**

Run:

```bash
git -C /home/john/elspeth worktree add /home/john/elspeth/.worktrees/tutorial-assumption-surfacing RC5.2
cd /home/john/elspeth/.worktrees/tutorial-assumption-surfacing
```

Expected: new worktree on `RC5.2`.

- [ ] **Step 2: Check status and preserve unrelated files**

Run:

```bash
git status --short --branch
```

Expected: clean worktree. If using the main checkout instead, do not touch unrelated untracked audit files under `docs/audit/`.

- [ ] **Step 3: Run the focused baseline tests**

Run:

```bash
uv run pytest tests/unit/contracts/test_composer_interpretation.py tests/unit/web/test_interpretation_state.py tests/unit/web/composer/test_request_interpretation_review_tool.py tests/unit/web/sessions/test_interpretation_events_table.py -q
```

Expected: PASS before edits.

## Task 1: Add InterpretationKind And Session Schema Epoch

**Files:**

- Modify: `src/elspeth/contracts/composer_interpretation.py`
- Modify: `src/elspeth/web/sessions/models.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/sessions/routes/_helpers.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Test: `tests/unit/contracts/test_composer_interpretation.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_table.py`
- Test: `tests/unit/web/sessions/test_interpretation_schemas.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_service.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_routes.py`

- [ ] **Step 1: Write contract tests for the enum and V2 domain**

Add assertions in `tests/unit/contracts/test_composer_interpretation.py`:

```python
from elspeth.contracts.composer_interpretation import (
    INTERPRETATION_HASH_DOMAIN_V2,
    InterpretationKind,
)


def test_interpretation_kind_closed_set() -> None:
    assert [member.value for member in InterpretationKind] == [
        "vague_term",
        "invented_source",
        "llm_prompt_template",
    ]


def test_v2_hash_domain_includes_kind_and_retires_v1_writes() -> None:
    assert "kind" in INTERPRETATION_HASH_DOMAIN_V2
    assert INTERPRETATION_HASH_DOMAIN_V2 == frozenset(
        {
            "session_id",
            "composition_state_id",
            "affected_node_id",
            "tool_call_id",
            "user_term",
            "kind",
            "llm_draft",
            "accepted_value",
            "actor",
            "model_identifier",
            "model_version",
            "provider",
            "composer_skill_hash",
        }
    )
```

Change existing `"v1"` assertions for resolved surface rows to `"v2"`. Keep tests for marker rows asserting both `arguments_hash` and `hash_domain_version` are `None`.

- [ ] **Step 2: Run contract tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/contracts/test_composer_interpretation.py -q
```

Expected: FAIL because `InterpretationKind` and V2 do not exist.

- [ ] **Step 3: Implement the L0 contract**

In `src/elspeth/contracts/composer_interpretation.py`, add:

```python
class InterpretationKind(StrEnum):
    """Class of LLM-authored assumption surfaced for review.

    CLOSED LIST - adding a value requires contract amendment, schema update,
    closed-enum tests, and writer-path audit.
    """

    VAGUE_TERM = "vague_term"
    INVENTED_SOURCE = "invented_source"
    LLM_PROMPT_TEMPLATE = "llm_prompt_template"
```

Add the dataclass field after `user_term`:

```python
    kind: InterpretationKind | None
```

Update enum validation:

```python
        if self.kind is not None:
            _validate_enum_member(self.kind, InterpretationKind, "kind")
```

Replace the active hash domain:

```python
INTERPRETATION_HASH_DOMAIN_V2: frozenset[str] = frozenset(
    {
        "session_id",
        "composition_state_id",
        "affected_node_id",
        "tool_call_id",
        "user_term",
        "kind",
        "llm_draft",
        "accepted_value",
        "actor",
        "model_identifier",
        "model_version",
        "provider",
        "composer_skill_hash",
    }
)
```

Retain `INTERPRETATION_HASH_DOMAIN_V1` only for read/history comments if needed; no writer may use it.

- [ ] **Step 4: Update session schema tests for the new column and shapes**

In `tests/unit/web/sessions/test_interpretation_events_table.py`, add `"kind"` to expected columns and add three insert-shape tests:

```python
def test_surface_opt_out_row_carries_kind_and_hash(engine) -> None:
    session_id = _insert_session(engine)
    state_id = _insert_state(engine, session_id)
    row_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            insert(interpretation_events_table).values(
                id=row_id,
                session_id=session_id,
                composition_state_id=state_id,
                affected_node_id="source",
                tool_call_id="call-1",
                user_term="inline_source_url_list",
                kind="invented_source",
                llm_draft="https://example.gov.au",
                accepted_value="https://example.gov.au",
                choice="opted_out",
                created_at=_now(),
                resolved_at=_now(),
                actor="composer-llm",
                model_identifier="test-model",
                model_version="test-model-20260524",
                provider="test",
                composer_skill_hash="a" * 64,
                arguments_hash="b" * 64,
                hash_domain_version="v2",
                interpretation_source="auto_interpreted_opt_out",
                runtime_model_identifier_at_resolve=None,
                runtime_model_version_at_resolve=None,
                resolved_prompt_template_hash=None,
            )
        )
```

Also assert session-level marker rows still allow `kind=None` and `accepted_value=None`.

- [ ] **Step 5: Run schema tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/web/sessions/test_interpretation_events_table.py -q
```

Expected: FAIL on missing `kind` column and old CHECK constraints.

- [ ] **Step 6: Bump the session schema epoch and update table constraints**

In `src/elspeth/web/sessions/models.py`:

```python
#   10 -> interpretation_events.kind added; surface-specific
#         AUTO_INTERPRETED_OPT_OUT rows now carry reviewed LLM artefacts and
#         V2 argument hashes.
SESSION_SCHEMA_EPOCH = 10
```

Add the column after `user_term`:

```python
    Column("kind", String, nullable=True),
```

Add the closed enum CHECK:

```python
    CheckConstraint(
        "kind IS NULL OR kind IN ('vague_term', 'invented_source', 'llm_prompt_template')",
        name="ck_interpretation_events_kind",
    ),
```

Replace the opt-out nullability CHECK with two explicit shapes:

```python
    CheckConstraint(
        "(interpretation_source != 'auto_interpreted_opt_out') OR "
        "((kind IS NULL AND composition_state_id IS NULL AND affected_node_id IS NULL AND "
        "  tool_call_id IS NULL AND user_term IS NULL AND llm_draft IS NULL AND "
        "  accepted_value IS NULL AND model_identifier IS NULL AND model_version IS NULL AND "
        "  provider IS NULL AND composer_skill_hash IS NULL AND arguments_hash IS NULL AND "
        "  hash_domain_version IS NULL) OR "
        " (kind IS NOT NULL AND composition_state_id IS NOT NULL AND affected_node_id IS NOT NULL AND "
        "  tool_call_id IS NOT NULL AND user_term IS NOT NULL AND llm_draft IS NOT NULL AND "
        "  accepted_value IS NOT NULL AND model_identifier IS NOT NULL AND model_version IS NOT NULL AND "
        "  provider IS NOT NULL AND composer_skill_hash IS NOT NULL AND arguments_hash IS NOT NULL AND "
        "  hash_domain_version = 'v2'))",
        name="ck_interpretation_events_opt_out_shape",
    ),
```

Update accepted-value constraint so surface opt-out rows may carry `accepted_value`:

```python
    CheckConstraint(
        "((choice IN ('accepted_as_drafted', 'amended')) OR "
        " (interpretation_source = 'auto_interpreted_opt_out' AND kind IS NOT NULL)) "
        "= (accepted_value IS NOT NULL)",
        name="ck_interpretation_events_accepted_value_status",
    ),
```

- [ ] **Step 7: Hydrate and expose kind**

In `_interpretation_event_record_from_row`, add:

```python
        kind=InterpretationKind(row.kind) if row.kind is not None else None,
```

In `InterpretationEventResponse`, add:

```python
    kind: Literal["vague_term", "invented_source", "llm_prompt_template"] | None = None
```

In `_interpretation_event_response`, add:

```python
        kind=event.kind.value if event.kind is not None else None,
```

Update `SessionServiceProtocol` method docs/signatures where pending and no-surface writers mention row shape.

- [ ] **Step 8: Update service hash construction to V2**

In `resolve_interpretation_event`, change the domain dict to include kind:

```python
domain_dict = {
    "session_id": sid,
    "composition_state_id": surfacing_state_id_str,
    "affected_node_id": event_row.affected_node_id,
    "tool_call_id": event_row.tool_call_id,
    "user_term": event_row.user_term,
    "kind": event_row.kind,
    "llm_draft": event_row.llm_draft,
    "accepted_value": accepted_value,
    "actor": actor,
    "model_identifier": event_row.model_identifier,
    "model_version": event_row.model_version,
    "provider": event_row.provider,
    "composer_skill_hash": event_row.composer_skill_hash,
}
if set(domain_dict.keys()) != INTERPRETATION_HASH_DOMAIN_V2:
    raise AssertionError("interpretation hash domain fields drifted")
arguments_hash = stable_hash(domain_dict)
```

Write `hash_domain_version="v2"`.

- [ ] **Step 9: Run contract and schema tests**

Run:

```bash
uv run pytest tests/unit/contracts/test_composer_interpretation.py tests/unit/web/sessions/test_interpretation_events_table.py tests/unit/web/sessions/test_interpretation_schemas.py -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/elspeth/contracts/composer_interpretation.py src/elspeth/web/sessions/models.py src/elspeth/web/sessions/service.py src/elspeth/web/sessions/schemas.py src/elspeth/web/sessions/routes/_helpers.py src/elspeth/web/sessions/protocol.py tests/unit/contracts/test_composer_interpretation.py tests/unit/web/sessions/test_interpretation_events_table.py tests/unit/web/sessions/test_interpretation_schemas.py
git commit -m "feat: add interpretation kind contract"
```

## Task 2: Make Composer-Authored Source Provenance Mechanical

**Files:**

- Modify: `src/elspeth/web/composer/tools/_common.py`
- Modify: `src/elspeth/web/composer/tools/_dispatch.py`
- Modify: `src/elspeth/web/composer/service.py`
- Modify: `src/elspeth/web/composer/tools/blobs.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/interpretation_state.py`
- Test: `tests/unit/web/composer/test_tools.py`
- Test: `tests/unit/web/composer/test_promote_set_pipeline.py`
- Test: `tests/unit/web/composer/test_promote_create_blob.py`
- Test: `tests/unit/web/test_interpretation_state.py`

- [ ] **Step 1: Write provenance tests**

In `tests/unit/web/composer/test_promote_create_blob.py`, add:

```python
def test_create_blob_marks_llm_generated_when_content_not_in_user_message(make_tool_context) -> None:
    ctx = make_tool_context(
        data_dir="/tmp/data",
        session_engine=_session_engine(),
        session_id=str(uuid4()),
        user_message_id=str(uuid4()),
        user_message_content="Please invent five government URLs.",
        composer_model_identifier="anthropic/claude-opus-4-7",
        composer_model_version="anthropic/claude-opus-4-7-20260101",
        composer_provider="anthropic",
        composer_skill_hash="a" * 64,
        tool_arguments_hash="b" * 64,
    )
    result = _execute_create_blob(
        {"filename": "urls.json", "mime_type": "application/json", "content": "[\"https://example.gov.au\"]"},
        _empty_state(),
        ctx,
    )
    assert result.success is True
    blob = result.data["blob"]
    assert blob["creation_modality"] == "llm_generated"
    assert blob["creating_model_identifier"] == "anthropic/claude-opus-4-7"
```

Add a paired test where `content` is contained in `user_message_content`; assert `creation_modality == "verbatim"` and `creating_model_identifier is None`.

- [ ] **Step 2: Run provenance tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/web/composer/test_promote_create_blob.py::test_create_blob_marks_llm_generated_when_content_not_in_user_message -q
```

Expected: FAIL because `ToolContext` lacks the provenance fields and `create_blob` always writes `verbatim`.

- [ ] **Step 3: Extend ToolContext and execute_tool**

In `ToolContext`, add:

```python
    user_message_content: str | None = None
    composer_model_identifier: str | None = None
    composer_model_version: str | None = None
    composer_provider: str | None = None
    composer_skill_hash: str | None = None
    tool_arguments_hash: str | None = None
```

Add the same keyword parameters to the `execute_tool` test/helper signature and pass them into `ToolContext`.

In `ComposerServiceImpl._dispatch_tool_batch`, pass:

```python
user_message_content=message,
composer_model_identifier=self._model,
composer_model_version=safe_response_model(response) or self._model,
composer_provider=self._availability.provider or "unknown",
composer_skill_hash=self._composer_skill_hash,
tool_arguments_hash=audit.arguments_hash,
```

- [ ] **Step 4: Add a provenance helper in `blobs.py`**

Add:

```python
@dataclass(frozen=True, slots=True)
class _BlobCreationProvenance:
    creation_modality: CreationModality
    creating_model_identifier: str | None
    creating_model_version: str | None
    creating_provider: str | None
    creating_composer_skill_hash: str | None
    creating_arguments_hash: str | None


def _blob_creation_provenance(content: str, context: ToolContext) -> _BlobCreationProvenance:
    user_text = context.user_message_content
    if user_text is not None and content in user_text:
        return _BlobCreationProvenance(
            creation_modality=CreationModality.VERBATIM,
            creating_model_identifier=None,
            creating_model_version=None,
            creating_provider=None,
            creating_composer_skill_hash=None,
            creating_arguments_hash=None,
        )
    required = {
        "composer_model_identifier": context.composer_model_identifier,
        "composer_model_version": context.composer_model_version,
        "composer_provider": context.composer_provider,
        "composer_skill_hash": context.composer_skill_hash,
        "tool_arguments_hash": context.tool_arguments_hash,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise AuditIntegrityError(f"LLM-generated blob provenance missing fields: {missing!r}")
    return _BlobCreationProvenance(
        creation_modality=CreationModality.LLM_GENERATED,
        creating_model_identifier=context.composer_model_identifier,
        creating_model_version=context.composer_model_version,
        creating_provider=context.composer_provider,
        creating_composer_skill_hash=context.composer_skill_hash,
        creating_arguments_hash=context.tool_arguments_hash,
    )
```

Use this helper in `_execute_create_blob` and `_execute_set_pipeline` inline-blob path.

- [ ] **Step 5: Stamp source authoring metadata**

In `src/elspeth/web/interpretation_state.py`, add:

```python
SOURCE_AUTHORING_KEY = "source_authoring"
SOURCE_COMPONENT_ID = "source"

class SourceAuthoringMetadata(TypedDict):
    modality: str
    content_hash: str
    review_event_id: str | None
    resolved_kind: str | None

AUTHORING_METADATA_OPTION_KEYS = frozenset(
    {
        INTERPRETATION_REQUIREMENTS_KEY,
        PROMPT_TEMPLATE_PARTS_KEY,
        SOURCE_AUTHORING_KEY,
    }
)
```

When `_execute_set_pipeline` creates an inline blob or binds a blob whose `creation_modality` is LLM-authored, add to `src_options`:

```python
SOURCE_AUTHORING_KEY: {
    "modality": prepared_inline_blob.creation_modality.value,
    "content_hash": prepared_inline_blob.content_hash,
    "review_event_id": None,
    "resolved_kind": None,
}
```

For `set_source_from_blob`, use the resolved blob's `creation_modality` and `content_hash`.

- [ ] **Step 6: Run provenance and YAML tests**

Run:

```bash
uv run pytest tests/unit/web/composer/test_promote_create_blob.py tests/unit/web/composer/test_promote_set_pipeline.py tests/unit/web/composer/test_yaml_generator.py tests/unit/web/test_interpretation_state.py -q
```

Expected: PASS. `test_yaml_generator.py` must prove `source_authoring` does not appear in generated YAML.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/composer/tools/_common.py src/elspeth/web/composer/tools/_dispatch.py src/elspeth/web/composer/service.py src/elspeth/web/composer/tools/blobs.py src/elspeth/web/composer/tools/sessions.py src/elspeth/web/interpretation_state.py tests/unit/web/composer/test_promote_create_blob.py tests/unit/web/composer/test_promote_set_pipeline.py tests/unit/web/composer/test_tools.py tests/unit/web/test_interpretation_state.py
git commit -m "feat: record composer-authored source provenance"
```

## Task 3: Extend Structured Interpretation State Per Kind

**Files:**

- Modify: `src/elspeth/web/interpretation_state.py`
- Modify: `src/elspeth/web/execution/service.py`
- Modify: `src/elspeth/web/execution/validation.py`
- Test: `tests/unit/web/test_interpretation_state.py`
- Test: `tests/unit/web/execution/test_service.py`
- Test: `tests/unit/web/execution/test_validation.py`

- [ ] **Step 1: Write kind-aware interpretation-state tests**

In `tests/unit/web/test_interpretation_state.py`, add:

```python
def test_pending_invented_source_requirement_blocks_execution() -> None:
    state = CompositionState(
        source=SourceSpec(
            plugin="json",
            on_success="rows",
            on_validation_failure="fail",
            options={
                SOURCE_AUTHORING_KEY: {"modality": "llm_generated", "content_hash": "a" * 64, "review_event_id": None, "resolved_kind": None},
                INTERPRETATION_REQUIREMENTS_KEY: [
                    {
                        "id": "source-urls",
                        "kind": "invented_source",
                        "user_term": "inline_source_url_list",
                        "status": "pending",
                        "draft": "https://example.gov.au",
                        "event_id": None,
                        "accepted_value": None,
                        "accepted_artifact_hash": None,
                        "resolved_prompt_template_hash": None,
                    }
                ],
            },
        ),
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    result = materialize_state_for_execution(state)
    assert isinstance(result, InterpretationReviewPending)
    assert result.sites[0].kind is InterpretationKind.INVENTED_SOURCE
    assert result.sites[0].component_id == "source"
```

Add tests for:

- pending `LLM_PROMPT_TEMPLATE` on an `llm` node
- resolved `LLM_PROMPT_TEMPLATE` requiring `stable_hash(prompt_template)`
- missing resolved Class 3 requirement on an `llm` node with non-empty `prompt_template`
- source metadata with `modality="llm_generated"` but no resolved Class 2 requirement

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/web/test_interpretation_state.py -q
```

Expected: FAIL because sites are still `(node_id, term)` tuples and source requirements are not walked.

- [ ] **Step 3: Add review-site dataclass and requirement fields**

In `interpretation_state.py`, import `Sequence` from `collections.abc`, then add:

```python
@dataclass(frozen=True, slots=True)
class InterpretationReviewSite:
    component_id: str
    component_type: Literal["source", "transform"]
    user_term: str
    kind: InterpretationKind


class InterpretationRequirement(TypedDict):
    id: str
    kind: str
    user_term: str
    status: Literal["pending", "resolved"]
    draft: str | None
    event_id: str | None
    accepted_value: str | None
    accepted_artifact_hash: str | None
    resolved_prompt_template_hash: str | None
```

Change `InterpretationReviewPending.sites` to:

```python
    sites: Sequence[InterpretationReviewSite]
```

- [ ] **Step 4: Replace `interpretation_sites(nodes)` call sites**

Add:

```python
def interpretation_sites(state: CompositionState) -> Sequence[InterpretationReviewSite]:
    sites: list[InterpretationReviewSite] = []
    if state.source is not None:
        sites.extend(_pending_source_sites(state.source.options))
    for node in state.nodes:
        sites.extend(_pending_node_sites(node))
        if node.plugin == "llm":
            sites.extend(_legacy_placeholder_sites(node))
            sites.extend(_missing_prompt_template_review_sites(node))
    return tuple(dict.fromkeys(sites))
```

Update callers:

- `materialize_state_for_execution(state)`
- `_detect_unresolved_interpretation_placeholders_typed` in `tools/sessions.py` or move its detection to the new state-level helper
- execution validation/service blocker formatting

- [ ] **Step 5: Implement per-kind materialization**

For source:

```python
def _materialize_source_for_execution(source: SourceSpec) -> SourceSpec:
    metadata = _source_authoring_metadata(source.options)
    requirements = _requirements(source.options)
    if metadata is None or requirements is None:
        return source
    resolved = _resolved_requirement_for_kind(requirements, InterpretationKind.INVENTED_SOURCE)
    if resolved is None:
        return source
    accepted_hash = resolved["accepted_artifact_hash"]
    if accepted_hash != metadata["content_hash"]:
        raise ValueError("invented source review drift: reviewed content hash does not match current source content hash")
    return source
```

For prompt templates:

```python
def _validate_prompt_template_review(node: NodeSpec, requirements: Sequence[InterpretationRequirement]) -> None:
    prompt_template = node.options.get("prompt_template")
    if not isinstance(prompt_template, str) or not prompt_template:
        return
    resolved = _resolved_requirement_for_kind(requirements, InterpretationKind.LLM_PROMPT_TEMPLATE)
    if resolved is None:
        raise ValueError(f"llm node {node.id!r} has prompt_template with no resolved prompt-template review")
    expected_hash = stable_hash(prompt_template)
    if resolved["resolved_prompt_template_hash"] != expected_hash:
        raise ValueError(f"llm node {node.id!r} prompt-template review hash drifted")
```

Keep vague-term rendering through `prompt_template_parts`.

- [ ] **Step 6: Update execution blocker formatting**

In validation/service paths, turn sites into messages:

```python
for site in pending.sites:
    component_type = site.component_type
    code = INTERPRETATION_REVIEW_PENDING_CODE
    message = f"{site.kind.value} review pending for {component_type} {site.component_id!r}: {site.user_term}"
```

Use component type `"source"` for Class 2 and `"transform"` for Class 1/Class 3.

- [ ] **Step 7: Run execution tests**

Run:

```bash
uv run pytest tests/unit/web/test_interpretation_state.py tests/unit/web/execution/test_service.py tests/unit/web/execution/test_validation.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/interpretation_state.py src/elspeth/web/execution/service.py src/elspeth/web/execution/validation.py tests/unit/web/test_interpretation_state.py tests/unit/web/execution/test_service.py tests/unit/web/execution/test_validation.py
git commit -m "feat: enforce kind-aware interpretation readiness"
```

## Task 4: Make request_interpretation_review Kind-Aware

**Files:**

- Modify: `src/elspeth/web/composer/tools/_dispatch.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/composer/redaction.py`
- Modify: `tests/unit/web/composer/redaction_policy_snapshot.json`
- Modify: `tests/unit/web/composer/test_request_interpretation_review_tool.py`
- Modify: `tests/unit/web/composer/test_request_interpretation_review_redaction.py`
- Modify: `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`
- Modify: `tests/unit/web/composer/test_tool_declarations.py`
- Modify: `tests/unit/web/composer/test_tools.py`

- [ ] **Step 1: Write tool schema and handler tests**

In `tests/unit/web/composer/test_request_interpretation_review_tool.py`, add:

```python
async def test_request_interpretation_review_accepts_prompt_template_kind(service, state_with_llm_prompt_template) -> None:
    result = await _handle_request_interpretation_review(
        {
            "affected_node_id": "identify_colour",
            "kind": "llm_prompt_template",
            "user_term": "llm_prompt_template:identify_colour",
            "llm_draft": "Read {{ row.html }} and return JSON.",
        },
        state_with_llm_prompt_template,
        **_provenance_kwargs(service),
    )
    assert result.success is True
    assert result.data["kind"] == "llm_prompt_template"
```

Add a negative test proving `llm_prompt_template` may contain `{{ row.html }}` while `vague_term` rejects template metacharacters.

Add a source test:

```python
async def test_request_interpretation_review_accepts_source_component_for_invented_source(service, state_with_llm_generated_source) -> None:
    result = await _handle_request_interpretation_review(
        {
            "affected_node_id": "source",
            "kind": "invented_source",
            "user_term": "inline_source_url_list",
            "llm_draft": "https://example.gov.au",
        },
        state_with_llm_generated_source,
        **_provenance_kwargs(service),
    )
    assert result.success is True
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/web/composer/test_request_interpretation_review_tool.py -q
```

Expected: FAIL because `kind` is extra and source is not accepted.

- [ ] **Step 3: Update inline LLM-facing schema**

In `_REQUEST_INTERPRETATION_REVIEW_DEFINITION` in `_dispatch.py`:

```python
"required": ["affected_node_id", "kind", "user_term", "llm_draft"],
"properties": {
    "affected_node_id": {
        "type": "string",
        "description": "Component id. Use 'source' for invented source data; use the LLM node id for vague terms and prompt templates.",
    },
    "kind": {
        "type": "string",
        "enum": ["vague_term", "invented_source", "llm_prompt_template"],
        "description": "Class of assumption being surfaced for review.",
    },
    "user_term": {
        "type": "string",
        "description": "Stable user-facing label for the assumption being reviewed.",
    },
    "llm_draft": {
        "type": "string",
        "description": "LLM-authored interpretation, source data, or prompt template text.",
    },
}
```

Update the description prose to say this is session-aware and kind-tagged.

- [ ] **Step 4: Update the session-aware argument model and validation**

In `tools/sessions.py`:

```python
class _RequestInterpretationReviewArgumentsModel(BaseModel):
    affected_node_id: str = Field(min_length=1, max_length=256)
    kind: InterpretationKind
    user_term: str = Field(min_length=1, max_length=8192)
    llm_draft: str = Field(min_length=1, max_length=8192)
```

Replace `_assert_affected_llm_node` with:

```python
def _assert_affected_component(
    state: CompositionState,
    affected_node_id: str,
    kind: InterpretationKind,
    user_term: str,
) -> None:
    if kind is InterpretationKind.INVENTED_SOURCE:
        if affected_node_id != SOURCE_COMPONENT_ID:
            raise ToolArgumentError(argument="affected_node_id", expected="'source' for invented_source", actual_type="node id")
        if state.source is None or SOURCE_AUTHORING_KEY not in state.source.options:
            raise ToolArgumentError(argument="affected_node_id", expected="source with composer-authored source metadata", actual_type="source without metadata")
        return
    node = _find_node_or_raise(state, affected_node_id)
    if node.plugin != "llm":
        raise ToolArgumentError(argument="affected_node_id", expected="id of an llm node", actual_type=f"plugin={node.plugin!r}")
    # Vague-term requirements may be structured or legacy placeholders.
    # Prompt-template requirements require a non-empty prompt_template string.
```

Apply `_validate_accepted_value_content(parsed.llm_draft)` only for `VAGUE_TERM` and `INVENTED_SOURCE`. For `LLM_PROMPT_TEMPLATE`, run only `_reject_credential_shaped_content` plus byte/shape checks.

- [ ] **Step 5: Pass kind into service writers and ToolResult payloads**

Call:

```python
event = await create_pending_interpretation_event(
    affected_node_id=parsed.affected_node_id,
    user_term=parsed.user_term,
    llm_draft=parsed.llm_draft,
    kind=parsed.kind,
)
```

Return:

```python
data={
    "_kind": "interpretation_review_pending",
    "event_id": str(event.id),
    "affected_node_id": parsed.affected_node_id,
    "kind": parsed.kind.value,
    "user_term": parsed.user_term,
    "llm_draft": parsed.llm_draft,
    "interpretation_source": event.interpretation_source.value,
}
```

- [ ] **Step 6: Update redaction model and snapshot**

In `redaction.py`:

```python
class _RequestInterpretationReviewRedactionModel(BaseModel):
    affected_node_id: str
    kind: InterpretationKind
    user_term: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]
    llm_draft: Annotated[str, Sensitive(summarizer=_summarize_interpretation_term)]
```

Regenerate snapshot:

```bash
uv run python scripts/cicd/bootstrap_redaction_snapshot.py --write
```

Expected: `tests/unit/web/composer/redaction_policy_snapshot.json` changes only for `request_interpretation_review`.

- [ ] **Step 7: Run composer tool tests**

Run:

```bash
uv run pytest tests/unit/web/composer/test_request_interpretation_review_tool.py tests/unit/web/composer/test_request_interpretation_review_redaction.py tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py tests/unit/web/composer/test_tool_declarations.py tests/unit/web/composer/test_tools.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/composer/tools/_dispatch.py src/elspeth/web/composer/tools/sessions.py src/elspeth/web/composer/redaction.py tests/unit/web/composer/redaction_policy_snapshot.json tests/unit/web/composer/test_request_interpretation_review_tool.py tests/unit/web/composer/test_request_interpretation_review_redaction.py tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py tests/unit/web/composer/test_tool_declarations.py tests/unit/web/composer/test_tools.py
git commit -m "feat: surface interpretation review kinds through composer"
```

## Task 5: Resolve And Opt-Out Rows Per Kind

**Files:**

- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/protocol.py`
- Modify: `src/elspeth/web/sessions/routes/interpretation.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_service.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_routes.py`
- Test: `tests/integration/web/composer/test_interpretation_runtime_handoff.py`
- Create: `tests/integration/web/test_interpretation_opt_out_audit.py`

- [ ] **Step 1: Write service tests for three resolve paths**

In `tests/unit/web/sessions/test_interpretation_events_service.py`, add:

```python
async def test_resolve_prompt_template_review_records_hash_without_rewriting_template(service) -> None:
    event = await _create_prompt_template_interpretation_event(
        service,
        kind=InterpretationKind.LLM_PROMPT_TEMPLATE,
    )
    resolved, new_state = await service.resolve_interpretation_event(
        session_id=event.session_id,
        event_id=event.id,
        choice=InterpretationChoice.ACCEPTED_AS_DRAFTED,
        amended_value=None,
        actor="user:alice",
    )
    assert resolved.kind is InterpretationKind.LLM_PROMPT_TEMPLATE
    assert resolved.hash_domain_version == "v2"
    node = next(node for node in new_state.nodes if node["id"] == "identify_colour")
    assert node["options"]["prompt_template"] == event.llm_draft
    assert node["options"]["resolved_prompt_template_hash"] == stable_hash(event.llm_draft)
```

Add a source Class 2 test that resolves without changing source content and stamps the source requirement with `event_id`, `accepted_value`, and `accepted_artifact_hash`.

- [ ] **Step 2: Write opt-out audit integration test**

In `tests/integration/web/test_interpretation_opt_out_audit.py`:

```python
async def test_opted_out_session_still_records_surface_specific_rows(composer_service, sessions_service) -> None:
    session_id = await _seed_opted_out_session(sessions_service)
    result = await _run_compose_with_three_request_interpretation_review_calls(composer_service, session_id)
    events = await sessions_service.list_interpretation_events(session_id, status="all")
    surface_opt_outs = [
        event for event in events
        if event.interpretation_source is InterpretationSource.AUTO_INTERPRETED_OPT_OUT and event.kind is not None
    ]
    assert [event.kind for event in surface_opt_outs] == [
        InterpretationKind.INVENTED_SOURCE,
        InterpretationKind.VAGUE_TERM,
        InterpretationKind.LLM_PROMPT_TEMPLATE,
    ]
    assert all(event.accepted_value is not None for event in surface_opt_outs)
```

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/web/sessions/test_interpretation_events_service.py tests/integration/web/test_interpretation_opt_out_audit.py -q
```

Expected: FAIL because service writers are not kind-aware.

- [ ] **Step 4: Update `create_pending_interpretation_event`**

Add `kind: InterpretationKind` parameter. For normal pending rows, insert:

```python
kind=kind.value,
interpretation_source=InterpretationSource.USER_APPROVED.value,
```

When `interpretation_review_disabled` is true, insert a new surface-specific opt-out row instead of returning the existing session marker:

```python
domain_dict = _interpretation_hash_domain_v2(
    session_id=sid,
    composition_state_id=state_id_str,
    affected_node_id=affected_node_id,
    tool_call_id=tool_call_id,
    user_term=user_term,
    kind=kind,
    llm_draft=llm_draft,
    accepted_value=llm_draft,
    actor="composer-llm",
    model_identifier=model_identifier,
    model_version=model_version,
    provider=provider,
    composer_skill_hash=composer_skill_hash,
)
arguments_hash = stable_hash(domain_dict)
```

For `LLM_PROMPT_TEMPLATE`, set `resolved_prompt_template_hash=stable_hash(llm_draft)`.

- [ ] **Step 5: Branch resolve logic by kind**

In `resolve_interpretation_event` after `accepted_value` validation:

```python
kind = InterpretationKind(event_row.kind)
if kind is InterpretationKind.VAGUE_TERM:
    final_source, final_nodes, resolved_prompt_template_hash = _resolve_vague_term(event_row, accepted_value)
elif kind is InterpretationKind.LLM_PROMPT_TEMPLATE:
    final_source, final_nodes, resolved_prompt_template_hash = _resolve_prompt_template(event_row, accepted_value)
elif kind is InterpretationKind.INVENTED_SOURCE:
    final_source, final_nodes, resolved_prompt_template_hash = _resolve_invented_source(event_row, accepted_value)
else:
    assert_never(kind)
```

Rules:

- `VAGUE_TERM`: preserve current prompt-patching behavior.
- `LLM_PROMPT_TEMPLATE`: require `accepted_value == current prompt_template`; stamp hash, update requirement status, do not rewrite prompt text.
- `INVENTED_SOURCE`: require current source authoring `content_hash` matches the requirement's draft artefact hash; update requirement status on source options, do not mutate source payload.

- [ ] **Step 6: Keep route validation scoped**

`InterpretationResolveRequest` remains `accepted_as_drafted | amended`. For Class 2/Class 3, the frontend will hide amend, but the backend should reject amended Class 2/Class 3 until the deferred C2 path exists:

```python
if kind in {InterpretationKind.INVENTED_SOURCE, InterpretationKind.LLM_PROMPT_TEMPLATE} and choice is InterpretationChoice.AMENDED:
    raise ValueError(f"{kind.value} does not support inline amendment in this release")
```

- [ ] **Step 7: Run session tests**

Run:

```bash
uv run pytest tests/unit/web/sessions/test_interpretation_events_service.py tests/unit/web/sessions/test_interpretation_events_routes.py tests/integration/web/composer/test_interpretation_runtime_handoff.py tests/integration/web/test_interpretation_opt_out_audit.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/elspeth/web/sessions/service.py src/elspeth/web/sessions/protocol.py src/elspeth/web/sessions/routes/interpretation.py tests/unit/web/sessions/test_interpretation_events_service.py tests/unit/web/sessions/test_interpretation_events_routes.py tests/integration/web/composer/test_interpretation_runtime_handoff.py tests/integration/web/test_interpretation_opt_out_audit.py
git commit -m "feat: resolve interpretation events by kind"
```

## Task 6: Update Composer Skill And Canonical Prompt

**Files:**

- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`
- Modify: `src/elspeth/web/preferences/tutorial_cache.py`
- Modify: `src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx`
- Test: `tests/unit/web/preferences/test_tutorial_cache.py`
- Test: `tests/unit/web/composer/test_tutorial_service.py`
- Test: `src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx`

- [ ] **Step 1: Update canonical prompt tests**

In backend and frontend prompt tests, assert:

```text
Please go to the following web pages, use abuse contact noreply@dta.gov.au
and scraping reason 'DTA technical demonstration'. Read the HTML for each
page, have an LLM identify the primary colours for each government agency.
Remove the HTML and save the rest to a json file.
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
uv run pytest tests/unit/web/preferences/test_tutorial_cache.py tests/unit/web/composer/test_tutorial_service.py -q
npm test -- --run src/components/chat/ChatPanel.test.tsx
```

Expected: FAIL on old prompt strings.

- [ ] **Step 3: Replace prompt constants and comments**

Update:

```typescript
export const CANONICAL_TUTORIAL_PROMPT =
  "Please go to the following web pages, use abuse contact noreply@dta.gov.au\n" +
  "and scraping reason 'DTA technical demonstration'. Read the HTML for each\n" +
  "page, have an LLM identify the primary colours for each government agency.\n" +
  "Remove the HTML and save the rest to a json file.";
```

Use the same byte-identical string in `tutorial_cache.py`.

- [ ] **Step 4: Replace skill assumption guidance**

In `pipeline_composer.md`, replace the chat-narration-only source guidance and subjective-term-only guidance with:

```markdown
**Surface every assumption you make.** Three classes of LLM-authored content must be surfaced for the user via `request_interpretation_review` before the pipeline can run. Every call carries `kind` matching `InterpretationKind` from `elspeth.contracts.composer_interpretation`. When `interpretation_review_disabled=true`, the call still emits an audit event (`interpretation_source=AUTO_INTERPRETED_OPT_OUT`) but no human-review surface renders. Opt-out skips the human, not the audit.

1. **Vague-term interpretations** (`kind="vague_term"`) - when the user prompt contains a subjective or underspecified term, surface your definition.
2. **Invented source data** (`kind="invented_source"`) - if you invent source content the user did not provide, call `request_interpretation_review` immediately after `set_source`, `set_source_from_blob`, `create_blob`, or `set_pipeline` succeeds. Use a stable `user_term` such as `inline_source_url_list`.
3. **LLM prompt templates** (`kind="llm_prompt_template"`) - every `prompt_template` you author for an `llm` transform must surface through `request_interpretation_review` with `user_term="llm_prompt_template:<node_id>"` and `llm_draft` equal to the raw template text.
```

- [ ] **Step 5: Run prompt and skill-adjacent tests**

Run:

```bash
uv run pytest tests/unit/web/preferences/test_tutorial_cache.py tests/unit/web/composer/test_tutorial_service.py tests/unit/web/composer/test_skill_drift.py -q
npm test -- --run src/components/chat/ChatPanel.test.tsx
```

Expected: PASS. If the skill hash changes, tests that assert `PIPELINE_COMPOSER_SKILL_HASH` must use the new hash produced by `load_skill_with_hash("pipeline_composer")`.

- [ ] **Step 6: Commit**

```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts src/elspeth/web/preferences/tutorial_cache.py src/elspeth/web/frontend/src/components/chat/ChatPanel.tsx tests/unit/web/preferences/test_tutorial_cache.py tests/unit/web/composer/test_tutorial_service.py src/elspeth/web/frontend/src/components/chat/ChatPanel.test.tsx
git commit -m "feat: update tutorial prompt and assumption guidance"
```

## Task 7: Build Kind-Aware Interpretation UI

**Files:**

- Modify: `src/elspeth/web/frontend/src/types/interpretation.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/hooks/useInterpretationResolver.ts`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx`
- Modify: `src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.test.tsx`
- Modify: `src/elspeth/web/frontend/src/types/interpretation.test.ts`

- [ ] **Step 1: Add frontend type tests**

In `src/elspeth/web/frontend/src/types/interpretation.test.ts`, assert sample event parsing includes:

```typescript
kind: "llm_prompt_template",
```

and rejects an unknown kind in any fixture validation helper already used by the file.

- [ ] **Step 2: Run type tests and confirm failure**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/types/interpretation.test.ts src/components/chat/guided/InterpretationReviewTurn.test.tsx
```

Expected: FAIL because `kind` is not typed and UI copy is not kind-aware.

- [ ] **Step 3: Add TypeScript kind union**

In `types/interpretation.ts`:

```typescript
export type InterpretationKind =
  | "vague_term"
  | "invented_source"
  | "llm_prompt_template";
```

Add `kind: InterpretationKind | null` to the existing `InterpretationEvent` interface without changing the existing timestamp, draft, status, and provenance fields.

- [ ] **Step 4: Extend `InterpretationReviewTurn` props**

Add:

```typescript
export interface InterpretationReviewTurnProps {
  event: InterpretationEvent;
  sessionId: string;
  showOptOut?: boolean;
  showAmend?: boolean;
  autoFocusOnMount?: boolean;
  onResolved?: (newState: CompositionState | null) => void;
}
```

Default `showOptOut = true`, `showAmend = true`, `autoFocusOnMount = true`.

- [ ] **Step 5: Add per-kind rendering helpers**

Add:

```typescript
function reviewCopy(event: InterpretationEvent): {
  heading: string;
  body: JSX.Element;
  draft: JSX.Element;
  requiresScroll: boolean;
} {
  switch (event.kind) {
    case "invented_source":
      return {
        heading: "Invented source data",
        body: <>You did not provide this source data. Review it before the pipeline fetches anything.</>,
        draft: <InventedSourceDraft value={event.llm_draft ?? ""} />,
        requiresScroll: false,
      };
    case "llm_prompt_template":
      return {
        heading: "LLM prompt template",
        body: <>This is the instruction written for {event.affected_node_id ?? "this transform"}.</>,
        draft: <PromptTemplateDraft value={event.llm_draft ?? ""} onScrolledToEnd={markTemplateScrolledToEnd} />,
        requiresScroll: true,
      };
    case "vague_term":
    case null:
      return {
        heading: "Interpretation review",
        body: <>When you said <em>{event.user_term ?? "this term"}</em>, I read that as roughly <em>{event.llm_draft ?? ""}</em>.</>,
        draft: <></>,
        requiresScroll: false,
      };
  }
}
```

Use an exhaustive `never` check for non-null future values.

- [ ] **Step 6: Add prompt-template scroll gate**

Disable the accept button while:

```typescript
const mustReadTemplate = event.kind === "llm_prompt_template";
const acceptDisabled = primaryButtonsDisabled || (mustReadTemplate && !templateScrolledToEnd);
```

Use:

```typescript
function onTemplateScroll(event: UIEvent<HTMLPreElement>): void {
  const target = event.currentTarget;
  const reachedEnd = target.scrollTop + target.clientHeight >= target.scrollHeight - 8;
  if (reachedEnd) setTemplateScrolledToEnd(true);
}
```

- [ ] **Step 7: Hide opt-out and amend by props**

Render amend button only when:

```typescript
showAmend && event.kind !== "invented_source" && event.kind !== "llm_prompt_template"
```

Render opt-out only when `showOptOut`.

- [ ] **Step 8: Run frontend unit tests**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/types/interpretation.test.ts src/components/chat/guided/InterpretationReviewTurn.test.tsx
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/web/frontend/src/types/interpretation.ts src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.tsx src/elspeth/web/frontend/src/hooks/useInterpretationResolver.ts src/elspeth/web/frontend/src/components/chat/guided/GuidedTurn.tsx src/elspeth/web/frontend/src/components/chat/guided/InterpretationReviewTurn.test.tsx src/elspeth/web/frontend/src/types/interpretation.test.ts
git commit -m "feat: render interpretation reviews by kind"
```

## Task 8: Stop Tutorial Auto-Opt-Out And Render All Assumptions

**Files:**

- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/copy.ts`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorial.css`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx`

- [ ] **Step 1: Write tutorial behavior tests**

In `TutorialTurn2Describe.test.tsx`, assert:

```typescript
expect(api.optOutOfInterpretations).not.toHaveBeenCalled();
expect(api.resolveInterpretation).not.toHaveBeenCalled();
```

Add a test where `fetchComposerPreferences` returns `{ interpretation_review_disabled: true }`; assert `buildTutorialDraft` rejects with `"tutorial sessions must not have interpretation review disabled"`.

In `TutorialTurn2bShowBuilt.test.tsx`, seed three pending events and assert:

```typescript
expect(screen.getByText("3 assumptions to review")).toBeInTheDocument();
expect(screen.getByRole("button", { name: /looks good/i })).toBeDisabled();
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/components/tutorial/TutorialTurn2Describe.test.tsx src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx src/components/tutorial/HelloWorldTutorial.test.tsx
```

Expected: FAIL because tutorial still opts out and resolves automatically.

- [ ] **Step 3: Remove auto opt-out and auto resolve**

In `buildTutorialDraft`, delete:

```typescript
await api.optOutOfInterpretations(session.id);
```

Delete the `resolveTutorialInterpretations` helper and call. After fetching preferences:

```typescript
if (composerPreferences?.interpretation_review_disabled === true) {
  throw new Error("tutorial sessions must not have interpretation review disabled");
}
```

Keep proposal auto-accept.

- [ ] **Step 4: Render every pending assumption**

In `TutorialTurn2bShowBuilt.tsx`:

```typescript
const pendingInterpretations = useMemo(() => {
  return Object.values(pendingBySession[sessionId] ?? {})
    .filter((event) => event.choice === "pending" && event.interpretation_source === "user_approved")
    .sort((a, b) => a.created_at.localeCompare(b.created_at));
}, [pendingBySession, sessionId]);

const hasPending = pendingInterpretations.length > 0;
```

Render stack:

```tsx
{pendingInterpretations.length > 1 && (
  <h3 className="tutorial-assumption-count">{pendingInterpretations.length} assumptions to review</h3>
)}
{pendingInterpretations.map((event, index) => (
  <InterpretationReviewTurn
    key={event.id}
    event={event}
    sessionId={sessionId}
    showOptOut={false}
    showAmend={event.kind === "vague_term"}
    autoFocusOnMount={index === 0}
    onResolved={(newState) => {
      if (newState !== null) useSessionStore.setState({ compositionState: newState });
    }}
  />
))}
```

Disable CTA:

```tsx
<button
  type="button"
  className="btn btn-primary"
  disabled={hasPending}
  aria-disabled={hasPending}
  title={hasPending ? "Approve the assumptions above first" : undefined}
  onClick={onContinue}
>
  {TURN_2B_PRIMARY_BUTTON}
</button>
```

- [ ] **Step 5: Update heading and copy**

Change Turn 2b heading to:

```tsx
Here is what the composer drafted - review its assumptions.
```

Add a live status:

```tsx
<p role="status" className="sr-only">
  {hasPending ? "Approve the assumptions above first" : "All assumptions approved; ready to run."}
</p>
```

- [ ] **Step 6: Run tutorial tests**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/components/tutorial/TutorialTurn2Describe.test.tsx src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx src/components/tutorial/HelloWorldTutorial.test.tsx
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.tsx src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.tsx src/elspeth/web/frontend/src/components/tutorial/copy.ts src/elspeth/web/frontend/src/components/tutorial/tutorial.css src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2Describe.test.tsx src/elspeth/web/frontend/src/components/tutorial/TutorialTurn2bShowBuilt.test.tsx src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx
git commit -m "feat: require tutorial assumption review"
```

## Task 9: Add Turn 7 Graduation And Split Completion Semantics

**Files:**

- Modify: `src/elspeth/web/frontend/src/stores/preferencesStore.ts`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.tsx`
- Create: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.tsx`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/copy.ts`
- Modify: `src/elspeth/web/frontend/src/components/tutorial/tutorial.css`
- Test: `src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts`
- Test: `src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx`
- Test: `src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx`
- Test: `src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts`

- [ ] **Step 1: Write reducer and preference tests**

In `tutorialMachine.test.ts`, assert:

```typescript
const modeState = tutorialStateAt("mode");
const graduationState = tutorialStateAt("graduation");

expect(tutorialReducer(modeState, { type: "finishMode" }).step).toBe("graduation");
expect(previousStep(graduationState)).toBe("mode");
```

In `HelloWorldTutorial.test.tsx`, assert Turn 6 PATCH body contains only `default_mode`, and Turn 7 CTA PATCH body contains only `tutorial_completed_at`.

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/components/tutorial/tutorialMachine.test.ts src/components/tutorial/HelloWorldTutorial.test.tsx
```

Expected: FAIL because `"graduation"` and split preference actions do not exist.

- [ ] **Step 3: Split preference store actions**

In `PreferencesState`, replace:

```typescript
markTutorialCompleted: (mode: ComposerMode) => Promise<void>;
```

with:

```typescript
saveTutorialMode: (mode: ComposerMode) => Promise<void>;
markTutorialGraduated: () => Promise<void>;
```

Implement:

```typescript
saveTutorialMode: async (mode) => {
  if (get().writing) return;
  const previous = get().defaultMode;
  set({ defaultMode: mode, writing: true, writeError: null });
  try {
    const payload = await updateUserComposerPreferences({ default_mode: mode });
    set({
      defaultMode: payload.default_mode,
      tutorialCompletedAt: payload.tutorial_completed_at,
      tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
      writing: false,
    });
  } catch (err) {
    set({ defaultMode: previous, writing: false, writeError: "Couldn't save your preference." });
    throw err;
  }
},
markTutorialGraduated: async () => {
  if (get().writing) return;
  const stamp = new Date().toISOString();
  const previous = { tutorialCompletedAt: get().tutorialCompletedAt, tutorialCompleted: get().tutorialCompleted };
  set({ writing: true, writeError: null });
  try {
    const payload = await updateUserComposerPreferences({ tutorial_completed_at: stamp });
    set({
      tutorialCompletedAt: payload.tutorial_completed_at,
      tutorialCompleted: tutorialCompletedFrom(payload.tutorial_completed_at),
      writing: false,
    });
  } catch (err) {
    set({
      tutorialCompletedAt: previous.tutorialCompletedAt,
      tutorialCompleted: previous.tutorialCompleted,
      writing: false,
      writeError: "Couldn't save tutorial completion.",
    });
    throw err;
  }
},
```

Update call sites.

- [ ] **Step 4: Extend tutorial state machine**

Add:

```typescript
| "graduation"
```

to `TutorialStep`, and:

```typescript
| { type: "finishMode" }
```

to `TutorialAction`.

Add reducer branch:

```typescript
case "finishMode":
  return nextTutorialState(state, "graduation");
```

Add previous-step branch:

```typescript
case "graduation":
  return "mode";
```

- [ ] **Step 5: Update Turn 6**

In `TutorialTurn6ModeChoice`, replace:

```typescript
await usePreferencesStore.getState().markTutorialCompleted(mode);
await useSessionStore.getState().createSession();
onFinished?.();
```

with:

```typescript
await usePreferencesStore.getState().saveTutorialMode(mode);
onFinished?.();
```

Keep session rename on Turn 6.

- [ ] **Step 6: Create Turn 7**

Create `TutorialTurn7Graduation.tsx`:

```tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { usePreferencesStore } from "@/stores/preferencesStore";
import { useSessionStore } from "@/stores/sessionStore";

interface TutorialTurn7GraduationProps {
  onBack: () => void;
}

export function TutorialTurn7Graduation({ onBack }: TutorialTurn7GraduationProps): JSX.Element {
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    headingRef.current?.focus();
    window.dispatchEvent(new CustomEvent("tutorial_graduation_shown"));
  }, []);

  const onFinish = useCallback(async () => {
    setPending(true);
    setError(null);
    try {
      await usePreferencesStore.getState().markTutorialGraduated();
      await useSessionStore.getState().createSession();
    } catch (err) {
      setError(err instanceof Error ? err.message : "The tutorial could not be completed.");
    } finally {
      setPending(false);
    }
  }, []);

  return (
    <section className="tutorial-turn" aria-labelledby="tutorial-graduation-title">
      <p className="tutorial-kicker">Graduation</p>
      <h2 id="tutorial-graduation-title" ref={headingRef} tabIndex={-1}>
        You're ready to use the composer.
      </h2>
      <ul className="tutorial-graduation-list">
        <li><strong>What you built is AI-generated.</strong> The pipeline you just ran was authored by an LLM that interpreted your one-sentence description.</li>
        <li><strong>Read before you run.</strong> Glance at the graph and YAML before clicking Run.</li>
        <li><strong>Ask Elspeth.</strong> If anything in a pipeline does not make sense, ask in the chat panel.</li>
        <li><strong>LLMs are confident even when they're wrong.</strong> Verify sources and check the output matches what you asked for.</li>
      </ul>
      <div className="tutorial-actions">
        <button type="button" className="btn btn-primary" disabled={pending} onClick={() => void onFinish()}>
          {pending ? "Saving" : "Take me to the composer"}
        </button>
        <button type="button" className="tutorial-link-button" disabled={pending} onClick={onBack}>
          Back
        </button>
      </div>
      {error !== null && <p role="alert" className="tutorial-error">{error}</p>}
    </section>
  );
}
```

If the codebase has a telemetry helper for tutorial events, call that instead of the `CustomEvent`; keep the test asserting one `tutorial_graduation_shown` emission.

- [ ] **Step 7: Wire HelloWorldTutorial**

Add progress label `{ key: "graduation", label: "Ready" }`, add `stepIndex("graduation")`, and render Turn 7:

```tsx
{state.step === "graduation" && (
  <TutorialTurn7Graduation onBack={goBack} />
)}
```

Pass `onFinished={() => dispatch({ type: "finishMode" })}` to Turn 6.

- [ ] **Step 8: Run frontend tests**

Run:

```bash
cd src/elspeth/web/frontend
npm test -- --run src/components/tutorial/tutorialMachine.test.ts src/components/tutorial/TutorialTurn7Graduation.test.tsx src/components/tutorial/HelloWorldTutorial.test.tsx
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/elspeth/web/frontend/src/stores/preferencesStore.ts src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.ts src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.tsx src/elspeth/web/frontend/src/components/tutorial/TutorialTurn6ModeChoice.tsx src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.tsx src/elspeth/web/frontend/src/components/tutorial/copy.ts src/elspeth/web/frontend/src/components/tutorial/tutorial.css src/elspeth/web/frontend/src/components/tutorial/tutorialMachine.test.ts src/elspeth/web/frontend/src/components/tutorial/TutorialTurn7Graduation.test.tsx src/elspeth/web/frontend/src/components/tutorial/HelloWorldTutorial.test.tsx
git commit -m "feat: add tutorial graduation turn"
```

## Task 10: End-To-End Gates And Release Notes

**Files:**

- Modify: `src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts`
- Modify: `tests/integration/web/test_tutorial_routes.py`
- Modify: `tests/integration/web/test_preflight_per_class.py`
- Modify: `docs/superpowers/specs/2026-05-22-tutorial-assumption-surfacing-and-graduation-design.md` only if the operator wants the spec corrected after implementation.

- [ ] **Step 1: Add preflight integration tests**

Create `tests/integration/web/test_preflight_per_class.py`:

```python
async def test_preflight_rejects_unreviewed_llm_prompt_template(execution_service, session_with_llm_prompt) -> None:
    with pytest.raises(UnresolvedInterpretationPlaceholderError, match="llm_prompt_template"):
        await execution_service.start_run(session_with_llm_prompt.session_id)


async def test_preflight_rejects_unreviewed_invented_source(execution_service, session_with_llm_generated_source) -> None:
    with pytest.raises(UnresolvedInterpretationPlaceholderError, match="invented_source"):
        await execution_service.start_run(session_with_llm_generated_source.session_id)
```

- [ ] **Step 2: Update Playwright tutorial happy path**

In `tutorial.spec.ts`, seed mocked API responses with three pending interpretation events:

```typescript
const pendingEvents = [
  inventedSourceEvent,
  vagueTermEvent,
  promptTemplateEvent,
];
```

Assert:

```typescript
await expect(page.getByText("3 assumptions to review")).toBeVisible();
await expect(page.getByRole("button", { name: /looks good/i })).toBeDisabled();
```

Resolve all three, then assert Turn 6 PATCH and Turn 7 PATCH are separate.

- [ ] **Step 3: Run full targeted matrix**

Run backend:

```bash
uv run pytest tests/unit/contracts/test_composer_interpretation.py tests/unit/web/test_interpretation_state.py tests/unit/web/composer/test_request_interpretation_review_tool.py tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py tests/unit/web/sessions/test_interpretation_events_service.py tests/unit/web/sessions/test_interpretation_events_routes.py tests/integration/web/test_preflight_per_class.py tests/integration/web/test_interpretation_opt_out_audit.py tests/integration/web/test_tutorial_routes.py -q
```

Run frontend:

```bash
cd src/elspeth/web/frontend
npm run test
npm run build
```

Run lint and type gates:

```bash
uv run ruff check src/elspeth tests
uv run mypy src/elspeth
env PYTHONPATH=elspeth-lints/src .venv/bin/python -m elspeth_lints.core.cli check --rules trust_tier.tier_model --root src/elspeth
```

Expected: all commands exit 0.

- [ ] **Step 4: Manual tutorial verification**

Run the frontend/backend locally or against staging. Use the canonical tutorial prompt and verify:

- Turn 2 does not call `POST /interpretations/opt_out`.
- Turn 2b shows three assumption cards: invented source data, vague-term interpretation for primary colour, and LLM prompt template.
- The run CTA remains disabled until all three cards resolve.
- Turn 6 saves only `default_mode`.
- Turn 7 emits `tutorial_graduation_shown` and saves only `tutorial_completed_at`.

- [ ] **Step 5: Staging deploy note**

Because this changes Python backend code and the session DB schema epoch, staging needs:

```bash
systemctl status elspeth-web.service --no-pager --lines=40
```

Then delete/recreate the pre-release session DB using the project's existing operator runbook, restart `elspeth-web.service`, and verify:

```bash
curl --unix-socket /run/elspeth/uvicorn.sock -fsS http://localhost/api/health
curl -fsS https://elspeth.foundryside.dev/api/health
```

If Codex sandbox blocks systemd or sudo, report the exact blocker and the local artifact verification.

- [ ] **Step 6: Commit final integration changes**

```bash
git add src/elspeth/web/frontend/tests/e2e/tutorial.spec.ts tests/integration/web/test_tutorial_routes.py tests/integration/web/test_preflight_per_class.py tests/integration/web/test_interpretation_opt_out_audit.py
git commit -m "test: cover tutorial assumption review flow"
```

## Self-Review Checklist

- [ ] `InterpretationKind` is persisted, exposed on the API, and consumed by the frontend.
- [ ] `request_interpretation_review` remains session-aware async; no normal `ToolDeclaration` was added for it.
- [ ] The session schema epoch is bumped and stale DB startup failure is expected.
- [ ] `create_blob` and `set_pipeline.source.inline_blob` no longer fabricate `verbatim` for LLM-authored content.
- [ ] Class 2 source approval gates execution before network egress.
- [ ] Class 3 prompt-template approval gates LLM execution and records a stable hash.
- [ ] Opt-out skips human review but emits surface-specific audit rows.
- [ ] Tutorial no longer opts out or auto-resolves interpretations.
- [ ] Turn 7, not Turn 6, records `tutorial_completed_at`.
- [ ] YAML generation strips all web-only metadata.
- [ ] No logger/structlog row-level decision logging was added.

## Recommended Execution Order

Use `superpowers:subagent-driven-development` with one subagent per task after Task 0. Tasks 1-5 are backend-critical and should be reviewed before frontend work starts. Tasks 7-9 can run in parallel after Task 1 exposes `kind` on the API, but Task 8 should not be merged until Task 5 makes pending/resolved rows behave correctly.
