# Prompt: Sync Sub-Plans with Round 4 Spec Delta

## Context

The Web UX Composer MVP specs underwent a Round 4 expert panel review. The review
produced 4 critical fixes, 5 high-priority amendments, and a new seam contracts
document. All fixes have been applied to the **specs** already. Each sub-plan has
a "Round 4 Review Amendments" appendix noting which fixes apply to it.

**Your job:** Update the **body** of each sub-plan so that task descriptions,
steps, file maps, acceptance criteria, and code examples are consistent with the
updated specs. The appendix sections are pointers — the actual plan content needs
to match.

## Source of Truth

Read these files first — they contain the authoritative post-review state:

1. **Seam contracts:** `docs/superpowers/specs/2026-03-28-web-ux-seam-contracts.md`
2. **Sub-spec 1:** `docs/superpowers/specs/2026-03-28-web-ux-sub1-foundation-design.md`
3. **Sub-spec 2:** `docs/superpowers/specs/2026-03-28-web-ux-sub2-auth-sessions-design.md`
4. **Sub-spec 3:** `docs/superpowers/specs/2026-03-28-web-ux-sub3-catalog-design.md`
5. **Sub-spec 4:** `docs/superpowers/specs/2026-03-28-web-ux-sub4-composer-design.md`
6. **Sub-spec 5:** `docs/superpowers/specs/2026-03-28-web-ux-sub5-execution-design.md`
7. **Sub-spec 6:** `docs/superpowers/specs/2026-03-28-web-ux-sub6-frontend-design.md`

## Delta Summary — What Changed in the Specs

### C1/B8: Async/sync bridging (Sub-5)
- `_run_pipeline()` must use `_call_async()` helper (`asyncio.run_coroutine_threadsafe`) for all SessionService calls
- `ExecutionServiceImpl` constructor now accepts a `loop: asyncio.AbstractEventLoop` parameter (same loop as ProgressBroadcaster)
- New acceptance criterion 17a tests Run status transitions through real async SessionService
- New B8 fix section in the Thread Safety documentation

### C2: CompositionState.from_dict() (Sub-4)
- New `from_dict()` class methods on: `CompositionState`, `SourceSpec`, `NodeSpec`, `EdgeSpec`, `OutputSpec`, `PipelineMetadata`
- Round-trip invariant: `CompositionState.from_dict(s.to_dict()) == s`
- New acceptance criteria 17-18
- Sub-2's `from_record()` calls `CompositionState.from_dict()` to reconstruct domain objects from DB JSON

### C3/S1: landscape_url removed (Sub-4)
- `PipelineMetadata` no longer has a `landscape_url` field
- YAML generator no longer emits a `landscape` key — landscape URL comes from WebSettings at execution time
- `set_metadata` tool can only set `name` and `description`

### C3/S2: Source path allowlist (Sub-4 + Sub-5)
- `set_source()` tool validates that `path`/`file` options resolve under `{data_dir}/uploads/`
- `validate_pipeline()` now takes `(state, settings)` and has a new step 1: path allowlist check using `Path.resolve()` + `is_relative_to()`
- Defense in depth: checked at both composition-time and validation-time

### C4/S3: Secret key hard crash (Sub-1 + Sub-2)
- `WebSettings.secret_key` field has S3 annotation
- W16 guard upgraded from `log warning` to `raise SystemExit` in non-test environments

### H1: compose() signature (Sub-4)
- Changed from `compose(message, session, state)` to `compose(message, messages, state)`
- `messages: list[ChatMessageRecord]` — pre-fetched by route handler
- ComposerService does NOT depend on SessionService
- Route handler mediates: calls `get_messages()`, passes result to `compose()`

### H2: API wire key mapping (Sub-6)
- POST /messages response returns `{message, state}` — wire field is `state`
- Frontend store maps `response.state` → `compositionState` on destructure

### H3: Validation gate invariant (Sub-6)
- `clearValidation()` fires on ANY `compositionState.version` change
- `revertToVersion()` calls `clearValidation()` BEFORE updating compositionState
- `selectSession()` also calls `clearValidation()`

### H6: WebSocket close codes (Sub-5 + Sub-6)
- 1000 = normal closure (terminal state) → no reconnect, poll REST
- 1006 = abnormal → reconnect with backoff
- 1011 = internal error → no reconnect, poll REST
- 4001 = auth failure → no reconnect, logout

### H8p: Rate limiting (Sub-1 + Sub-4)
- New WebSettings field: `composer_rate_limit_per_minute: int = 10`
- POST /messages route enforces per-user rate limit, returns 429 when exceeded

### M1: /state/yaml endpoint in Sub-2 (Sub-2)
- Added to Sub-2's endpoint inventory
- Route handler loads active CompositionState, calls `generate_yaml(state)`
- Returns `{"yaml": str}` or 404
- Stub returning 501 until Sub-4 implements YAML generator

### M2: Error envelope standardisation (all subs)
- All errors use `detail` (not `message`) as human-readable field
- Domain errors add `error_type` as machine-readable discriminator
- 409 RunAlreadyActiveError includes `error_type: "run_already_active"`

### M3: Singular/plural plugin_type (Sub-3)
- CatalogService protocol uses singular: `"source"`, `"transform"`, `"sink"`
- REST paths use plural: `sources`, `transforms`, `sinks`
- Route handler translates plural → singular

### H5: tool_calls JSON schema (Sub-4 + Sub-6)
- Stored as LiteLLM format: `[{id, type, function: {name, arguments}}]`
- `arguments` is a JSON string (not parsed object)
- Frontend MessageBubble extracts `function.name` for display

### H7: Stage 1 vs Stage 2 error rendering (Sub-6)
- Stage 1: `string[]` from `compositionState.validation_errors` → simple list
- Stage 2: `ValidationError[]` from validate endpoint → per-component detail with attribution
- Separate renderer code paths, must NOT share a component

### M5: is_valid population (Sub-4)
- Route handler must call `state.validate()` before `save_composition_state()`
- Passes `is_valid` and `errors` to SessionService
- Without this, is_valid defaults to False and Execute button never enables

## Instructions

For each sub-plan (1 through 6):

1. **Read the sub-plan** in full.
2. **Read its corresponding spec** for the authoritative post-review state.
3. **Read the "Round 4 Review Amendments" appendix** at the end of the sub-plan — this tells you which fixes apply.
4. **Update the plan body** so that every task, step, code example, file map entry, and acceptance criterion is consistent with the spec. Specifically:

   - **Task descriptions** that reference changed signatures, fields, or behaviour must be updated inline (e.g., if a task says `compose(message, session, state)`, change it to `compose(message, messages, state)`)
   - **Code examples** that show removed fields (like `landscape_url`) or old patterns (like direct `session_service.update_run_status()` without `_call_async()`) must be updated
   - **File maps** that need new files (e.g., no new files were added, but check nothing was missed)
   - **Acceptance criteria** must match the spec's updated criteria
   - **Test descriptions** must cover the new behaviour (e.g., `test_state.py` must now include `from_dict()` round-trip tests)

5. **Do NOT delete the "Round 4 Review Amendments" appendix** — it stays as the traceability record. Add a note at the top: "Amendments below have been integrated into the plan body."
6. **Do NOT change the specs** — they are the source of truth. Only change the plans.

## Working Approach

Process the plans in dependency order: 1, then 2 and 3 (parallel-safe), then 4, then 5, then 6. For each plan, do a focused pass looking for every reference to changed items. Use grep/search for these terms to find stale references:

- `landscape_url` (removed from PipelineMetadata)
- `compose(message, session` (old signature)
- `session_service.update_run_status` without `_call_async` (in sub-5)
- `secret_key.*warning` or `log a warning` (now a crash)
- `message` as error field name (now `detail`)
- `_get_plugin_manager` (should be `get_shared_plugin_manager` after sub-1)

After updating each plan, verify the plan's internal consistency: do task numbers still flow correctly? Do cross-references between tasks still point to the right things?

## Deliverable

For each sub-plan modified, state:
- Number of inline changes made
- Which fix IDs were integrated
- Any inconsistencies found that could not be resolved from the spec alone (flag these for human review)
