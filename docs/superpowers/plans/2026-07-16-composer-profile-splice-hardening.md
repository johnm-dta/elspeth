# Composer Profile-Aware Splicing Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Let the composer insert a profile-backed Bedrock content guard into an existing linear pipeline in one atomic edit, validate the edit against the operator-resolved runtime contract, preserve only still-valid review evidence, and receive the configured compose budget in AWS.

**Architecture:** Keep `CompositionState` audit-safe and profile-aliased. Consolidate the three existing validation adapters behind one public profile-policy adapter, thread the already-authoritative catalog into it, and validate only its ephemeral `executable_state`. Establish authoritative review reconciliation before implementing topology edits, then add one narrow, idempotent `splice_transform` tool that validates and reconciles its complete candidate through the same boundaries as `upsert_node` before committing one state replacement.

**Tech stack:** Python 3.13, frozen dataclasses, Pydantic v2, JSON Schema tool declarations, pytest, React/TypeScript timeout bootstrap, Docker/ECR, AWS ECS/ALB/RDS, and a detached state-aligned copy of the acceptance Terraform package.

## Confirmed failure shape

- The live request received only 120 seconds because the active ECS task definition sets `ELSPETH_WEB__COMPOSER_TIMEOUT_SECONDS=120`; the intended acceptance value is 270 seconds.
- The browser adds 25 seconds to the backend value; the ALB idle timeout is 300 seconds. The intended ceilings are therefore backend 270, browser 295, ALB 300.
- The model tried repeated `set_pipeline` full replacements because no supported mutation can insert one transform and rewire both sides atomically.
- `aws_bedrock_prompt_shield` accepts public profile options, but raw `CompositionState.validate()` probes those public options without the existing runtime profile lowering and loses the shield's pass-through guarantees.
- `get_pipeline_state` exposes inspection and server-owned data that cannot be copied into the extra-forbid `SetPipelineArgumentsModel`.
- The final advisor call began with less than one second remaining. It could not produce useful guidance.
- The current live image already expects `run_sources.schema_contract_hash VARCHAR(32)` and readiness proves the live landscape is at that width, but the rebased target also adds cross-dialect landscape/session identity epochs, six PostgreSQL session audit triggers, the Landscape 23→26 ownership/idempotency/sink-effect chain, and new/rebuilt-table ACL obligations. The failed target-image rollout stopped at `session_schema`. A versioned migration must preflight width 16 or 32, never narrow it, preserve Tier-1 rows and runtime grants, and install the exact target cohort before startup. The current image cannot be restarted after the first session-epoch DDL.
- The checked-out acceptance Terraform package is materially dirty. Its changes include database-shape, listener, secret, and service changes outside this task, so it cannot be used as an apply source.

## Prerequisites and boundaries

- Implement in a fresh worktree based on `release/0.7.1`; do not edit the deployed release checkout.
- Prepare that worktree once with `uv sync --frozen --all-extras`. Do not patch around a base-only `.venv`.
- Preserve the current session database and audit rows. Do not rewrite diagnostic session `4fc43ba6-8a8c-468d-b55b-acdabc3d9831`.
- Run only the named focused regressions during Tasks 1-5. At the end of Task 5, pause exactly once for the operator to run the parallel full suite.
- Perform one whole-implementation review after Tasks 1-4B, not a review after every task.
- Do not add Wardline, signing, compatibility approvals, or CI ceremony. They do not contribute to this container correction.
- Use `AWS_PROFILE=elspeth-acceptance` and `AWS_REGION=ap-southeast-1` only in Task 6. If `aws sts get-caller-identity` reports an expired session, pause for the operator to run `aws login --profile elspeth-acceptance --region ap-southeast-1`.
- Before any AWS mutation, re-query the ECS deployment and readiness state. A read-only review observed another actor changing the service from `web:24` toward `web:25`; never overlap a live rollout.
- Do not edit, reset, clean, or apply the dirty acceptance Terraform package. Do not destroy or replace RDS/Aurora, ALB, EFS, Cognito, secrets, session storage, or audit storage.
- Do not regenerate the release trust-tier baseline. The original pre-Task-1 drift signature remains historical evidence at `/home/john/.local/state/elspeth/composer-splice-hardening/fingerprint-drift-pre-task1.json` with SHA-256 `e6eb0311ffff63534dff1430a51f5a1900b888ca07a7eeb57918f385e5f41485`. After the required rebase, the release branch itself carries later fingerprint rotation, so the final gate is an exact raw-finding pair comparison against the fetched `origin/release/0.7.1`, not a comparison to that stale pre-rebase artifact.

## Load-bearing implementation order

1. Make every composer validation path use the existing profile-aware policy validator.
2. Build the exact authoring snapshot and authoritative review-reconciliation seam.
3. Build the atomic splice on those two seams.
4. Teach the model to use the splice and reject useless late advisor calls.
5. Add the exact versioned schema migration required by the target image.
6. Review and verify the complete implementation, then pause for one operator-run full suite.
7. Build once, raise the timeout, deploy once, and repeat the original action live.

Across all tasks:

- Persisted/audited options retain only public profile aliases; private identifiers, regions, versions, endpoints, and credentials never cross authoring, tool-message, audit, log, or YAML boundaries.
- A failed mutation returns the original state object and version.
- A successful new splice advances the version exactly once. An identical retry is an idempotent no-op and does not advance it.
- Review evidence is trusted only from the authoritative previous state. Caller-supplied resolved metadata remains rejected.
- A resolved review is carried only when its identity and canonical artifact agree in the previous and proposed states. Unknown or incoherent review kinds fail closed.

## Review disposition ledger

The adjacent review remains the immutable input artifact. This plan supersedes its requested changes as follows.

| Review item | Disposition | Plan correction |
|---|---|---|
| B1 schema widening and rollback incompatibility | Accepted and expanded from live evidence | Task 4B adds a versioned/tested one-shot migration for the exact session identity/audit-trigger cohort and Landscape 23→26 chain, with idempotent 16-or-32 width preflight, atomic row-preserving table replacement, and ACL preservation. Task 6 runs it with the schema role before candidate startup, requires the candidate doctor to report `CURRENT`, never narrows, and uses forward recovery rather than an old-image restart. |
| B2 dirty/destructive Terraform and false idle-timeout proof | Accepted | Task 6 never applies the dirty package, reconstructs a detached state-aligned root, machine-rejects deletes/replacements/out-of-allowlist changes, and verifies the live ALB attribute directly. |
| B3 baseline drift test cannot distinguish plan drift | Accepted | Before rebase, the fixed pre-task artifact proved byte equality. After rebase, Task 5 scans the final branch and the exact fetched release tree with the same linter/empty allowlist and requires identical `(file_path, fingerprint)` sets. No baseline regeneration or rotation is allowed. |
| M1 `profile_unavailable` is unreachable | Accepted | Task 1 adds a closed public `error_code`, maps operator-profile failures to `profile_unavailable`, preserves enablement reasons, and carries the code through safe audit serialization. |
| M2 existing recipe expectations contradict the mapper | Accepted | Task 1 keeps the three exact recipe regressions and makes the generic mapper satisfy them. |
| M3 incomplete raw-validation inventory | Accepted | Task 1 covers prompts, emitters/generation, state summaries, routes, sessions, discovery, dispatcher, and batch bypasses; the intentional local trained-operator boundary is explicit. |
| M4 duplicate validation adapters | Accepted and broadened | Three existing adapters are consolidated; no fourth adapter is introduced. |
| M5 hot-path catalog construction | Accepted | Task 1 removes `create_catalog_service()` from validation and threads the authoritative application catalog through every direct caller. |
| M6 incomplete reconciliation matrix | Accepted | Task 2 dispatches hashes by `InterpretationKind` and `(kind, user_term)`, includes HTTP identity, preserves full row identity, and atomically structures unknown-term failures. |
| M7 empty aliases and missing registry | Accepted | Task 1 distinguishes intentional local trained-operator raw behavior from fail-closed web/user scope. |
| M8 selectors can select zero tests | Accepted | Tasks 1-5 use exact node IDs or whole dedicated files in one `&&` chain. |
| M9 fictional rollback/tag handling | Accepted; circuit-breaker subclaim rejected | Task 6 skips an already-existing immutable digest, uses task-definition facts only as evidence, and treats the schema migration as forward-only recovery. The live Terraform does have a circuit breaker, but its service ignores task-definition changes, so explicit `update-service` is required. |
| M10 browser/Cognito proof underspecified | Accepted | Task 6 names the protected OIDC credential source and ephemeral bearer handoff, uses the exact same-origin URL, and pauses only if that human-auth artifact is genuinely absent or unusable. |
| M11 unsafe deployment order | Accepted | Task 6 verifies current-image health first, migrates schema, runs the candidate doctor, then updates the service and timeout. |
| m1 `prompt-approved` fixture | Accepted with wording correction | Task 3 uses this plan's configured `prompt-default`; `prompt-approved` is not claimed to be globally invalid. |
| m2 audit field names | Accepted | Tasks 1 and 6 distinguish proposal `arguments_redacted_json`, invocation canonical JSON, and the LLM call's `tools_spec_hash`. |
| m3 resolver stub only raises `ValueError` | Accepted | Task 1 tests redaction and atomic failure for both `ValueError` and a non-`ValueError` exception. |
| m4 replay identity | Accepted | Task 3 asserts `is` identity and unchanged version. |
| m5 replay compared before review staging | Accepted | Task 3 deterministically normalizes both request and installed projections before comparison. |
| m6 deadline refusal discovery flag | Accepted | Task 4 sets `turn_has_discovery=True` while avoiding budget, LLM, and anti-anchor effects. |
| m7 all policy findings treated fail-closed | Accepted | Task 1 immediately blocks only enablement/profile findings; required-control findings remain incremental policy evidence. |
| m8 component/code carrier | Accepted | Task 1 defines the component mapping and audit-safe `error_code` carrier. |
| m9 direct lowering helper remains | Accepted | Task 1 deletes `lower_operator_profile_options()` after migrating its sole production caller. |
| m10 guided/steps validation path | Accepted | Task 1 removes the dead helper/re-export or routes any live path through the shared adapter; it is not silently omitted. |
| m11 exactly-five-second semantics | Accepted | Task 4 permits the call at exactly 5.0 seconds with effective timeout 5 and `advisor_deadline_limited=True`. |
| m12 timeout has zero headroom | Accepted | Task 6 intentionally pins and asserts 270/295/300 as an exact acceptance ceiling, rather than describing 270 as having headroom. |
| m13 opaque waits | Accepted | Task 6 uses bounded polling, ECS events, stopped-task reasons, and CloudWatch logs. |
| m14 redaction and tool-declaration proof | Accepted with schema correction | Task 3 extracts a shared narrow options model and tests a sentinel; Task 4 proves names and hash against the actual transmitted list, while Task 6 separately proves live declaration names and tool selection. |

Two literal reviewer suggestions are intentionally refined rather than copied: `_PipelineNodeModel` cannot be reused unchanged because splice derives required topology fields, so a shared narrow base/options model is extracted; and `prompt-approved` exists in other acceptance fixtures, while this plan correctly uses `prompt-default`.

---

### Task 1: Route composer validation through the existing profile-policy authority

**Files:**

- Modify: `src/elspeth/web/plugin_policy/validation.py`
- Modify: `src/elspeth/web/catalog/policy_view.py`
- Modify: `src/elspeth/web/composer/state.py`
- Modify: `src/elspeth/web/composer/guided/audit.py`
- Modify: `src/elspeth/web/composer/tools/_common.py`
- Modify: `src/elspeth/web/composer/tools/_dispatch.py`
- Modify: `src/elspeth/web/composer/tools/generation.py`
- Modify: `src/elspeth/web/composer/prompts.py`
- Modify: `src/elspeth/web/composer/discovery_cache.py`
- Modify: `src/elspeth/web/composer/tool_batch.py`
- Modify: `src/elspeth/web/composer/guided/emitters.py`
- Modify: `src/elspeth/web/composer/guided/steps.py` only to remove the dead step-4 helper/re-export, if its caller sweep remains empty
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/sessions/service.py`
- Modify: `src/elspeth/web/sessions/schemas.py`
- Modify: `src/elspeth/web/frontend/src/types/index.ts`
- Modify: `src/elspeth/web/sessions/routes/_helpers.py`
- Modify: `src/elspeth/web/sessions/routes/composer/guided.py`
- Modify: `src/elspeth/web/app.py` only as required to inject the existing application catalog into session validation
- Modify direct policy callers in `src/elspeth/web/audit_readiness/service.py`, `src/elspeth/web/composer/tutorial_service.py`, `src/elspeth/web/execution/service.py`, and `src/elspeth/web/execution/validation.py` as required by the catalog parameter
- Modify: `src/elspeth/web/aws_ecs_acceptance.py` so the live doctor/readiness path receives the authoritative catalog rather than constructing or omitting it
- Test: `tests/integration/web/test_bedrock_guardrail_policy.py`
- Test: `tests/unit/web/composer/test_tools.py`
- Test: `tests/unit/web/composer/test_prompts.py`
- Test: `tests/unit/web/composer/test_dispatch_arms_characterization.py`
- Test: `tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py`
- Test: `tests/unit/web/composer/test_request_interpretation_review_tool.py`
- Test: `tests/unit/web/composer/test_service.py`
- Test: `tests/unit/web/sessions/test_interpretation_events_service.py`
- Test: `tests/unit/web/sessions/test_schemas.py`
- Test: `tests/integration/web/composer/guided/test_wire_dispatch.py`
- Test: `tests/unit/composer_mcp/test_server.py`
- Test: `tests/unit/web/execution/test_validation.py`
- Test: `tests/integration/web/test_plugin_policy_end_to_end.py`
- Add: `tests/unit/web/composer/test_profile_policy_validation_adapter.py` for the new cross-seam matrix

**Step 1: Add a live-shaped failing contract regression**

Construct `PolicyCatalogView` with the existing authoritative catalog from `_policy_context()`, then author this path:

```text
web_scrape(page_text) -> aws_bedrock_prompt_shield(profile="prompt-default") -> llm(required_input_fields=["page_text"])
```

Assert the desired behavior and confirm it fails before implementation:

- the shield-to-LLM edge guarantees `page_text`;
- no computed-contract probe warning is emitted for the shield;
- the authored state still contains only `profile="prompt-default"`;
- preview and diff use the same contract result as execution policy validation.

**Step 2: Consolidate the validation authority and make its public result explicit**

Extend `validate_plugin_policy()` to accept the already-authoritative full catalog (or a narrow schema-provider protocol) rather than constructing one. Remove the `create_catalog_service()` import and call. `PolicyCatalogView.validate_authored_state()` passes its `_full` catalog; every direct caller listed above threads the application catalog it already owns. Session construction receives the app catalog once. No validation request creates a catalog service.

Introduce one shared profile-aware adapter around `validate_plugin_policy()` that returns all four values needed by existing consumers: the unchanged authored state, ephemeral executable state, public policy findings, and mapped `ValidationSummary`. Consolidate these three adapters on it:

- `SessionServiceImpl._validate_patched_composition_state`;
- routes `_wire_policy_validation`;
- guided `_guided_persisted_validity`.

The executable state is never serialized, persisted, logged, or returned to the model. The adapter maps policy findings to `ValidationEntry` with an optional closed `error_code`. Add that optional field to `ValidationEntry.to_dict()`, `ValidationEntryResponse`, every response-construction projection, the frontend `ValidationEntryDTO`, and the guided audit safe allowlist. Add a strict response/DTO regression so the classifier cannot be silently dropped. The public component mapping is exact:

- default source: `source`; named source: `source:<name>`;
- transform: `node:<id>`;
- sink: `output:<name>`;
- no component: `pipeline`.

Use the matching `PluginAvailability.reason.value` from the indexed `snapshot.unavailable` tuple for enablement failures; `PluginAvailabilitySnapshot` has no `unavailable_reason()` method. Every `operator_profile_options` failure uses `profile_unavailable`. Only `plugin_enablement` and `operator_profile_options` are immediate no-raw-fallback blockers. `required_control_availability` and `required_control_coverage` remain ordinary policy evidence so incremental authoring can build toward compliance.

**Step 3: Define web fail-closed and intentional local behavior**

Make principal scope explicit at the adapter boundary:

- `local:trained-operator` retains deliberate full-schema/raw validation and may have no profile registry;
- web/user scope fails closed for an empty alias registry only when the authored state requires a profile-backed plugin, and always for a missing registry/alias, stale alias, unsupported profile-bearing plugin, non-string alias, public-schema mismatch, or lowering failure. Unrelated CSV/passthrough/JSON pipelines remain valid without profile aliases.

Catch `Exception` only at the profile resolver boundary and convert it to generic `profile_unavailable`; do not log exception text or `exc_info`. Test both `ValueError("PRIVATE_SENTINEL_VALUE")` and `RuntimeError("PRIVATE_SENTINEL_RUNTIME")`.

Refactor `_prevalidate_transform_for_context` to build an ephemeral candidate state and invoke the shared adapter. Then delete `PolicyCatalogView.lower_operator_profile_options()` and its unused imports after its sole production caller is gone.

**Step 4: Route every behavior-affecting validation path**

Use the shared adapter or a previously computed result at all of these seams:

- prompts; guided emitters; generation, explain, preview, and diff;
- `_state_data_from_composer_state`, because it controls runtime-preflight execution;
- session patch/re-entry, routes helper, guided route, and the completed-profile re-entry;
- dispatcher success and failure normalization, including unknown-tool/argument/context failures;
- `_common` handler results, `tools/sessions.py`, discovery reconstruction/cache hits, approval proposals, and session-aware/batch bypasses.

Normalize a frozen `ToolResult` once at the final dispatch/batch boundary. Preserve the handler's leading `rejected_mutation` entry and public `error_code`; never replace it with an apparently valid summary. Any path that has no catalog context must receive it from the caller, not fall back to raw `state.validate()`.

The local MCP `composer_mcp/server.py` trained-operator boundary remains intentionally raw and gets a characterization test. Propagate the catalog signature through execution-service construction in `app.py`, readiness/tutorial APIs, the AWS doctor/acceptance path, and their execution/plugin-policy tests. Sweep `guided/steps.py::handle_step_4_wire_confirm`; if it remains dead, remove the helper, route re-export, and obsolete tests. If a live caller appears, migrate it instead.

**Step 5: Prove public classification, atomicity, and secrecy**

Add exact named tests for:

- `test_profiled_prompt_shield_contract_uses_executable_binding_without_mutating_authored_state`;
- `test_profile_validation_reuses_authoritative_catalog_without_construction`;
- `test_web_profile_validation_fails_closed_for_empty_alias_registry`;
- `test_web_profile_validation_fails_closed_when_registry_is_missing`;
- `test_profile_lowering_value_error_is_redacted_and_stable`;
- `test_profile_lowering_runtime_error_is_redacted_and_stable`;
- `test_profile_lowered_validation_preserves_prompt_shield_contract`;
- `test_rejected_mutation_validation_preserves_rejection_and_profile_code`;
- `test_required_control_coverage_does_not_block_incremental_authoring_validation`;
- `test_profile_validation_uses_executable_state_and_keeps_authored_alias`;
- `test_arm_discovery_cache_hit_uses_profile_aware_validation`;
- `test_explicit_approve_proposal_uses_profile_aware_validation`;
- `test_request_interpretation_review_result_uses_profile_aware_validation`;
- `test_state_data_from_composer_state_uses_profile_aware_authoring_validation`;
- `test_wire_profile_failure_carries_profile_unavailable_without_raw_fallback`;
- `test_completed_profiled_reentry_uses_profile_aware_validation`;
- `test_local_trained_operator_generate_yaml_keeps_raw_profile_behavior_without_registry`;
- `test_private_profile_binding_never_reaches_tool_message_audit_yaml_or_logs`.

Separate internal correctness from outward secrecy. For successful lowering, assert the private binding exists only in the ephemeral executable state and the authored state retains only the alias; do not serialize that executable state into an outward assertion. For resolver exceptions, assert the exception sentinel is absent even from the returned internal result. In both cases it must be absent from returned/persisted `ToolResult`, proposal `arguments_redacted_json`, persisted chat/API audit projection, safe response, generated YAML, guided audit evidence, and `caplog`. Keep the existing exact regressions:

- `test_apply_pipeline_recipe_reports_unavailable_profile_alias`;
- `test_apply_pipeline_recipe_reports_profile_unavailable_when_required_profile_is_omitted`;
- `test_apply_pipeline_recipe_preserves_disabled_llm_plugin_reason`;
- the two Bedrock policy tests, guided wire test, session revalidation test, local MCP characterization, prompt context test, cache-hit test, approval-proposal test, and interpretation-review dispatch tests named in the review evidence.

Run exact node IDs or whole dedicated files only; all commands are chained with `&&`. Update paths if a new dedicated test file is preferable, but never replace exact nodes with a broad `-k`.

Run this chain immediately after writing the RED tests and require the new nodes to fail for the intended missing behavior; rerun the identical chain GREEN before commit:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider \
  tests/unit/web/composer/test_profile_policy_validation_adapter.py \
  tests/integration/web/test_bedrock_guardrail_policy.py::test_profiled_prompt_shield_contract_uses_executable_binding_without_mutating_authored_state \
  tests/unit/web/composer/test_tools.py::test_apply_pipeline_recipe_reports_unavailable_profile_alias \
  tests/unit/web/composer/test_tools.py::test_apply_pipeline_recipe_reports_profile_unavailable_when_required_profile_is_omitted \
  tests/unit/web/composer/test_tools.py::test_apply_pipeline_recipe_preserves_disabled_llm_plugin_reason \
  tests/unit/web/sessions/test_schemas.py
```

**Step 6: Commit**

```bash
git add src/elspeth/web/plugin_policy/validation.py \
  src/elspeth/web/catalog/policy_view.py \
  src/elspeth/web/composer/state.py \
  src/elspeth/web/composer/guided/audit.py \
  src/elspeth/web/composer/tools/_common.py \
  src/elspeth/web/composer/tools/_dispatch.py \
  src/elspeth/web/composer/tools/generation.py \
  src/elspeth/web/composer/prompts.py \
  src/elspeth/web/composer/discovery_cache.py \
  src/elspeth/web/composer/tool_batch.py \
  src/elspeth/web/composer/guided/emitters.py \
  src/elspeth/web/composer/guided/steps.py \
  src/elspeth/web/composer/tools/sessions.py \
  src/elspeth/web/sessions/service.py \
  src/elspeth/web/sessions/schemas.py \
  src/elspeth/web/frontend/src/types/index.ts \
  src/elspeth/web/sessions/routes/_helpers.py \
  src/elspeth/web/sessions/routes/composer/guided.py \
  src/elspeth/web/app.py \
  src/elspeth/web/audit_readiness/service.py \
  src/elspeth/web/composer/tutorial_service.py \
  src/elspeth/web/execution/service.py \
  src/elspeth/web/execution/validation.py \
  src/elspeth/web/aws_ecs_acceptance.py \
  tests/integration/web/test_bedrock_guardrail_policy.py \
  tests/unit/web/composer/test_tools.py \
  tests/unit/web/composer/test_prompts.py \
  tests/unit/web/composer/test_dispatch_arms_characterization.py \
  tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py \
  tests/unit/web/composer/test_request_interpretation_review_tool.py \
  tests/unit/web/composer/test_service.py \
  tests/unit/web/sessions/test_interpretation_events_service.py \
  tests/unit/web/sessions/test_schemas.py \
  tests/integration/web/composer/guided/test_wire_dispatch.py \
  tests/unit/composer_mcp/test_server.py \
  tests/unit/web/execution/test_validation.py \
  tests/integration/web/test_plugin_policy_end_to_end.py \
  tests/unit/web/composer/test_profile_policy_validation_adapter.py
git commit -m "fix(composer): validate through operator profiles"
```

**Definition of done:** composer preview, diff, dispatch, and execution agree on the profile-lowered shield contract; mutation rejections remain visible; private bindings are absent from every persisted and returned surface tested above.

---

### Task 2: Create a true authoring payload and reconcile authoritative reviews

**Files:**

- Modify: `src/elspeth/web/composer/tools/_common.py`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/interpretation_state.py`
- Modify: `src/elspeth/web/sessions/service.py` where authoritative review rows are resolved
- Add: `tests/unit/web/composer/test_authoring_reconciliation.py`
- Test: `tests/unit/web/composer/test_tools.py`
- Test: `tests/unit/web/composer/test_promote_set_pipeline.py`
- Test: `tests/unit/web/composer/test_request_interpretation_review_tool.py`

**Step 1: Pin the current non-round-trippable state behavior**

Write desired-behavior tests and confirm they fail. Cover a state with resolved vague-term, prompt-template, model-choice, raw-cleanup, prompt-shield-recommendation, and invented-source reviews.

The tests must show:

- the ordinary full inspection view is not advertised as `set_pipeline` input;
- `component="set_pipeline_arguments"` returns a payload accepted exactly by `SetPipelineArgumentsModel`;
- server-owned resolution fields are absent from that payload;
- a default blob-backed source is represented as `source.blob_id`, with `path`, `blob_ref`, and `source_authoring` removed from caller-visible options;
- named/multiple blob-backed sources return a stable `round_trip_unavailable` result because v1 `set_pipeline.sources` cannot bind blobs safely;
- `version` and `inspection` are not inside the exact arguments payload.

Keep the existing inspection mode for diagnostics. Add the dedicated exact-arguments mode rather than pretending the inspection envelope itself is round-trippable.

**Step 2: Add the exact authoring serializer**

Implement a typed payload whose top-level keys are only:

```text
source | sources, nodes, edges, outputs, metadata
```

Make it structurally validate with `SetPipelineArgumentsModel.model_validate()` in tests. Strip top-level/runtime-owned hashes and resolver-owned requirement fields. Pending requirements remain authorable. Resolved `VAGUE_TERM` rows needed by `prompt_template_parts` are not simply deleted: the serializer emits a pending authoring shell with the same requirement ID, kind, `user_term`, and authoring draft, while omitting accepted/resolver-owned fields. Its `prompt_template_parts` reference remains intact, so no dangling `requirement_id` is created. The reconciliation helper may then reattach the authoritative resolved row when its canonical checks pass.

Only a single default source keyed `source` with one simple string `options.blob_ref` may emit `source.blob_id`. Strip `path`, `blob_ref`, `source_authoring`, and resolver metadata. Named sources, multiple sources, widened/non-string/missing blob identities, or any shape that `set_pipeline` cannot bind safely return structured `round_trip_unavailable`; do not emit a partial payload.

**Step 3: Add one authoritative review reconciliation helper**

Add a pure helper in `interpretation_state.py` used by both full replacement and splice:

```python
def reconcile_authoritative_reviews(
    previous: CompositionState,
    proposed: CompositionState,
) -> CompositionState:
    """Rehydrate only coherent, still-applicable server-owned review evidence."""
```

Rules:

1. Strip caller/proposed resolver-owned fields before reconciliation.
2. Match the full row identity: component type, component ID, plugin, requirement ID, `InterpretationKind`, and `user_term`.
3. Carry a resolved row only when its stored accepted hash equals the canonical previous artifact and that equals the canonical proposed artifact.
4. Dispatch canonical artifacts by the closed `InterpretationKind`, never by the misleading storage-field name:
   - `LLM_PROMPT_TEMPLATE`: `prompt_structure_hash(parts)` or the existing stable template hash;
   - `LLM_MODEL_CHOICE`: add/use `model_choice_artifact_hash(options.model)`, even though the persisted accepted hash currently occupies `resolved_prompt_template_hash`;
   - `PIPELINE_DECISION`: match by `(kind, user_term)` and use `pipeline_decision_artifact_hash()` for all three registered terms: raw HTML cleanup, prompt-injection-shield recommendation, and `WEB_SCRAPE_HTTP_IDENTITY_USER_TERM`;
   - `INVENTED_SOURCE`: `source_authoring.content_hash`;
   - `VAGUE_TERM`: preserve and re-render the existing requirement/reference `prompt_template_parts` semantics rather than reading a generic field.
5. Reject stale accepted hashes, incoherent resolved fields, duplicate identities, and unknown kinds. An unknown pipeline-decision `user_term` is deliberately a `ValueError` at the hash authority; both full replacement and splice catch reconciliation failure at their mutation boundary, return one stable closed error code, the exact previous state object/version, and no exception text.
6. Preserve unrelated coherent reviews.
7. When an artifact materially changes but the review is still required, explicitly stage a pending replacement; do not assume `_options_with_default_llm_reviews()` covers pipeline decisions or invented sources.
8. When inserting an effective prompt shield makes `prompt_injection_shield_recommendation` obsolete, remove that requirement rather than preserving or re-staging it.
9. Rehydrate `source_authoring` only from the authoritative previous default blob binding and only when blob/content identity still matches.
10. For resolved vague terms, preserve the pending authoring shell (`id`, `kind`, `user_term`, draft) and the same `prompt_template_parts` interpretation reference. Rehydrate an accepted row only when that reference and skeleton are unchanged, then render from parts. Legacy resolved rows without reconstructible parts/reference return `round_trip_unavailable`; never guess.

Call the helper in `_execute_set_pipeline` after the complete proposed topology exists and before `composition_review_contract_error()` and final validation.

**Step 4: Test the reconciliation matrix**

Table-drive every supported review kind and each registered pipeline-decision term through:

- unchanged artifact;
- relevant drift;
- unrelated drift;
- stale accepted hash;
- incoherent resolved metadata;
- duplicate requirement identity;
- unknown kind and unknown pipeline user term through both real handlers;
- pending and resolved prompt-shield recommendations before and after a shield appears.

Assert only the affected review changes. Explicitly test resolved vague-term references and invented-source/default-blob round trips. Caller-supplied resolver fields must still be rejected.

Put the matrix in `test_authoring_reconciliation.py` and run the whole file. Add exact existing-node regressions for the hash authorities and real `set_pipeline` handler. Do not use the former broad selectors: one selected zero tests and the others could pass on unrelated tests. Run this chain RED after adding the matrix, then GREEN before commit:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider \
  tests/unit/web/composer/test_authoring_reconciliation.py \
  tests/unit/web/test_interpretation_state.py::test_resolved_pipeline_decision_hash_drift_fails_closed \
  tests/unit/web/test_interpretation_state.py::test_web_scrape_http_identity_hash_treats_omitted_allowed_hosts_as_public_only \
  tests/unit/web/test_interpretation_state.py::test_pipeline_decision_artifact_hash_rejects_unknown_user_term
```

**Step 5: Commit**

```bash
git add src/elspeth/web/composer/tools/_common.py \
  src/elspeth/web/composer/tools/sessions.py \
  src/elspeth/web/interpretation_state.py \
  src/elspeth/web/sessions/service.py \
  tests/unit/web/composer/test_authoring_reconciliation.py \
  tests/unit/web/composer/test_tools.py \
  tests/unit/web/composer/test_promote_set_pipeline.py \
  tests/unit/web/composer/test_request_interpretation_review_tool.py
git commit -m "fix(composer): reconcile reviewed authoring state"
```

**Definition of done:** the dedicated payload is accepted by the real `set_pipeline` model; unsafe blob shapes fail closed; coherent unchanged decisions survive; material changes restage or retire only the relevant review.

---

### Task 3: Add an atomic, idempotent `splice_transform` mutation

**Files:**

- Modify: `src/elspeth/web/composer/redaction.py`
- Modify: `src/elspeth/web/composer/tools/transforms.py`
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md` (generated inventory block only)
- Modify generated snapshot: `tests/unit/web/composer/redaction_policy_snapshot.json`
- Add: `tests/unit/web/composer/test_splice_transform_tool.py`
- Add: `tests/integration/web/composer/test_profile_splice_flow.py`
- Add: `tests/testcontainer/web/test_composer_splice_concurrency.py`
- Test: `tests/unit/web/composer/test_adequacy_guard.py`
- Test: `tests/unit/web/composer/test_redact_tool_call_arguments.py`
- Test: `tests/unit/web/composer/test_redact_tool_call_response.py`
- Test: `tests/unit/web/composer/test_redaction_completeness_property.py`
- Test: `tests/unit/web/composer/test_tool_redaction_dataclass.py`
- Test: `tests/unit/web/composer/test_tool_redaction_policy.py`
- Test: `tests/unit/web/composer/test_tool_declarations.py`

**Step 1: Write behavior-level RED tests**

Declare the intended tool in the test through the public registry and assert it is currently unavailable, then add cases for:

1. `web_scrape -> llm` becomes `web_scrape -> prompt_shield -> llm`.
2. The new node is placed immediately before the successor in `state.nodes`.
3. The original visual edge is replaced in place for predecessor-to-new; the new-to-successor edge is adjacent.
4. One new splice advances the version once.
5. `test_splice_transform_identical_review_staged_replay_is_same_object` replays the identical normalized splice and asserts `already_applied=true`, `replay.updated_state is first.updated_state`, and no version change.
6. A same-ID but divergent retry fails with the original state/version.
7. Missing/non-direct/ambiguous paths, gates, forks, coalesces, queue branches, error routes, sinks, duplicate IDs, exhausted connection names, and edge-ID collisions fail atomically.
8. The profile-backed shield validates through Task 1.
9. The prompt-shield recommendation is retired, while unrelated prompt/model/raw-cleanup/vague-term reviews remain coherent through Task 2.
10. Two concurrent identical HTTP compose requests for the same session serialize through the in-process route lock, reload state between executions, produce one state-version increment, and return one applied result plus one idempotent result. A separate direct PostgreSQL persistence test expects one success plus one `StaleComposeStateError` under the advisory lock, with the winning state intact and no lost update; sequential replay remains the handler's identity/idempotence proof.

The public arguments are intentionally narrow:

```json
{
  "predecessor_id": "guided_xform_0",
  "successor_id": "guided_xform_1",
  "node": {
    "id": "prompt_shield",
    "plugin": "aws_bedrock_prompt_shield",
    "options": {
      "profile": "prompt-default",
      "fields": ["page_text"],
      "schema": {"mode": "observed"}
    },
    "on_error": "discard"
  }
}
```

The server derives `input`, `on_success`, and edge IDs from the direct path.

**Step 2: Extract a shared transform-candidate boundary**

Refactor `_execute_upsert_node` so a pure helper validates/prepares a transform candidate for both upsert and splice. It must retain every existing write-boundary rule:

- runtime/resolver-owned metadata rejection;
- plugin availability and profile policy;
- credential wiring;
- batch placement and required-input contracts;
- default prompt/model review staging;
- provider configuration and path policy;
- immediate candidate prevalidation delegated through `catalog.validate_authored_state()` as established in Task 1;
- `composition_review_contract_error()` after topology assembly.

Do not duplicate a subset in `splice_transform`. The helper returns a prepared `NodeSpec` or the existing structured failure without changing the state/version.

**Step 3: Implement the one-replacement splice**

The handler must:

1. Resolve a source/transform predecessor and transform successor.
2. Before requiring a fresh direct edge, normalize both the caller and installed authoring projections through the same deterministic defaults and review staging used for a new node, including `_options_with_default_llm_reviews()`. Then strip resolution-only values while preserving pending authoring identity. If that ID already exists, compare the normalized projections and require exact server-derived topology: predecessor routes to the inserted node, the inserted node routes directly to the successor, node/edge order is canonical, and no conflicting path exists. Return the identical state object/version with `already_applied=true`. A partial or divergent shape fails unchanged.
3. For a new splice, require exactly one direct visual `on_success` edge and matching `predecessor.on_success == successor.input`.
4. Reject structural branches and non-linear routes.
5. Derive a bounded connection name, checking all producers, `on_error`, routes, fork targets, queues/coalesces, and sink names. Try deterministic numeric suffixes within a fixed bound, then fail on exhaustion.
6. Prepare the new transform through the shared boundary.
7. Rewrite predecessor output, successor input, nodes, and edges in local immutable collections.
8. Run `reconcile_authoritative_reviews(previous, proposed)` on the complete topology.
9. Run the review contract and Task 1 context-aware validation.
10. Commit with one `dataclasses.replace(..., version=state.version + 1)`. Do not chain `with_node()` or `with_edge()`.
11. Return the affected nodes, structural diff, derived connection, and `already_applied` flag.

**Step 4: Add a redaction-bearing argument model**

The legacy declarative policy only sees top-level sensitive keys. `_PipelineNodeModel` cannot be reused unchanged because it requires `node_type` and `input`, which splice derives. Extract a shared narrow base/options model used by both `_PipelineNodeModel` and the splice node model. Preserve `extra="forbid"` and the same `Sensitive` structural summarizer for open options.

The persisted summary may expose only structural facts such as node ID, plugin, option-key names/count, predecessor, and successor. It must not expose option values. Unknown keys remain forbidden/redacted. Add `test_splice_transform_node_options_redaction_omits_values`: put a nested unique user-option sentinel in options, call `redact_tool_call_arguments`, and assert the sentinel is absent from `json.dumps(result)`. The raw in-memory `ComposerToolInvocation.arguments_canonical` and replay input intentionally retain exact user arguments and their hash invariant; do not claim they are redacted. Assert the sentinel is absent from proposal `arguments_redacted_json`, persisted chat/API audit projection, safe response serialization, YAML, and logs.

Append the declaration to `TOOLS_IN_MODULE`; do not add a manual shadow registry. Then regenerate the two governed artifacts:

```bash
uv run python scripts/cicd/generate_skill_inventory.py --write && \
uv run python scripts/cicd/bootstrap_redaction_snapshot.py --write && \
FIRST_SNAPSHOT_SHA="$(sha256sum tests/unit/web/composer/redaction_policy_snapshot.json | awk '{print $1}')" && \
uv run python scripts/cicd/bootstrap_redaction_snapshot.py --write && \
test "$FIRST_SNAPSHOT_SHA" = "$(sha256sum tests/unit/web/composer/redaction_policy_snapshot.json | awk '{print $1}')"
```

Do not modify either generator unless it has a real generic defect.

**Step 5: Add the exact combined live-failure regression**

In `test_profile_splice_flow.py`, start with the valid live-shaped `web_scrape -> llm` state and resolved review rows. Invoke `splice_transform` through the real dispatcher, then `preview_pipeline`.

Assert in one deterministic test:

- one version increment and correct node/edge ordering;
- `page_text` remains guaranteed through the shield;
- unrelated reviews remain resolved and the satisfied shield recommendation is gone;
- final state is valid;
- dispatcher, tool message, audit row, generated YAML, and logs contain no sentinel private binding.

This integration test proves the combined mechanics. The later live model run proves only tool selection and deployment behavior.

**Step 6: Run focused splice contracts**

```bash
uv run pytest tests/unit/web/composer/test_splice_transform_tool.py -q && \
uv run pytest tests/integration/web/composer/test_profile_splice_flow.py -q && \
uv run pytest -m 'testcontainer or not testcontainer' tests/testcontainer/web/test_composer_splice_concurrency.py -q && \
uv run pytest tests/unit/web/composer/test_adequacy_guard.py \
  tests/unit/web/composer/test_redact_tool_call_arguments.py \
  tests/unit/web/composer/test_redact_tool_call_response.py \
  tests/unit/web/composer/test_redaction_completeness_property.py \
  tests/unit/web/composer/test_tool_redaction_dataclass.py \
  tests/unit/web/composer/test_tool_redaction_policy.py \
  tests/unit/web/composer/test_tool_declarations.py -q && \
uv run python scripts/cicd/generate_skill_inventory.py --check
```

The adequacy guard is the check for the committed redaction snapshot; no nonexistent `test_redaction.py` path is used.

**Step 7: Commit**

```bash
git add src/elspeth/web/composer/redaction.py \
  src/elspeth/web/composer/tools/transforms.py \
  src/elspeth/web/composer/skills/pipeline_composer.md \
  tests/unit/web/composer/redaction_policy_snapshot.json \
  tests/unit/web/composer/test_splice_transform_tool.py \
  tests/integration/web/composer/test_profile_splice_flow.py \
  tests/testcontainer/web/test_composer_splice_concurrency.py \
  tests/unit/web/composer/test_adequacy_guard.py \
  tests/unit/web/composer/test_redact_tool_call_arguments.py \
  tests/unit/web/composer/test_redact_tool_call_response.py \
  tests/unit/web/composer/test_redaction_completeness_property.py \
  tests/unit/web/composer/test_tool_redaction_dataclass.py \
  tests/unit/web/composer/test_tool_redaction_policy.py \
  tests/unit/web/composer/test_tool_declarations.py
git commit -m "feat(composer): splice transforms atomically"
```

**Definition of done:** one audited tool call performs the complete linear insertion; replay is harmless; all failures are atomic; review evidence is reconciled; redaction/inventory gates include the new tool.

---

### Task 4: Teach the model the supported edit path and reject futile advisor calls

**Files:**

- Modify: `src/elspeth/contracts/composer_llm_audit.py`
- Modify: `src/elspeth/web/composer/llm_response_parsing.py`
- Modify: `src/elspeth/web/composer/audit.py`
- Modify: `src/elspeth/web/composer/skills/pipeline_composer.md`
- Modify: `src/elspeth/web/composer/tools/sessions.py`
- Modify: `src/elspeth/web/composer/tools/transforms.py`
- Modify: `src/elspeth/web/composer/anti_anchor.py`
- Modify: `src/elspeth/web/composer/tool_batch.py`
- Test: `tests/unit/web/composer/test_tool_declarations.py`
- Test: `tests/unit/web/composer/test_anti_anchor.py`
- Test: `tests/unit/web/composer/test_compose_loop_anti_anchor.py`
- Test: `tests/unit/web/composer/test_advisor_tool.py`

**Step 1: Pin tool-choice guidance**

Add failing assertions that:

- `set_pipeline` describes itself as create/full-rebuild only;
- `splice_transform` names insert/between/before/after on a direct linear path;
- the core skill selects `splice_transform` for one-node insertion, `patch_node_options` for option-only edits, and `set_pipeline` only for an intentional rebuild;
- `build_drift_hint()` generically says that, when the goal is one-node linear insertion, switch to `splice_transform` instead of varying full-replacement payloads.

Do not teach `AntiAnchorTracker` to infer intent it does not store, and do not add a mandatory advisor checkpoint.

**Step 2: Update descriptions and skill wording**

Replace the current `set_pipeline` efficiency claim with:

> Atomically create or fully rebuild a pipeline. For a narrow edit to an existing pipeline, use the dedicated patch tool; use `splice_transform` to insert one transform on a direct linear path.

Add the same concise decision table to the skill. Regenerate/check the inventory after hand-editing the non-generated guidance section.

**Step 3: Add an advisor remaining-time floor**

Define once in `tool_batch.py`:

```python
_MIN_USEFUL_ADVISOR_SECONDS: Final[float] = 5.0
```

After the existing `remaining <= 0` timeout branch and before timeout calculation or incrementing `advisor_calls_used`, reject `0 < remaining < 5.0` with a truthful successful payload:

```text
status=DEADLINE_TOO_CLOSE
outbound_call_made=false
```

The branch must set `turn_has_discovery=True`, but must not create a `ComposerLLMCall`, make an outbound call, consume advisor budget, record/reset anti-anchor progress, or fabricate guidance. Preserve `COMPOSE_TIMEOUT` for `remaining <= 0`. At exactly 5.0 seconds the call is allowed with `effective_timeout=5.0` and `advisor_deadline_limited=True`.

**Step 4: Test exact deadline boundaries**

Test remaining time at `0`, `4.999`, `5.0`, and just above `5.0`. Below the floor assert:

- zero outbound model calls and zero model-call audit rows;
- unchanged advisor budget;
- one truthful tool invocation;
- no guidance field containing fabricated advice;
- unchanged anti-anchor state.
- `turn_has_discovery=True` below the floor, matching other truthful policy refusals.

Add a public-safe `declared_tool_names: tuple[str, ...]` projection to the JSON `ComposerLLMCall` sidecar without adding a SQL table/column. Derive it from the exact transmitted list in `build_llm_call_record`, validate it as closed structural metadata, and expose it through `_LLM_CALL_PUBLIC_AUDIT_FIELDS`. Keep `tools_spec_hash` as the canonical integrity proof over the complete transmitted specification.

Add a declaration/audit test that captures the exact tools list passed to `_call_llm`, asserts `splice_transform` is declared, asserts sidecar `declared_tool_names` matches the transmitted names, and asserts `ComposerLLMCall.tools_spec_hash == stable_hash(transmitted_tools)` after any Anthropic cache-marker transformation. Tool selection remains separate `ComposerToolInvocation` evidence; declaration names never claim selection.

Run commands separately where filters differ:

```bash
uv run pytest tests/unit/web/composer/test_tool_declarations.py -q && \
uv run pytest tests/unit/web/composer/test_anti_anchor.py \
  tests/unit/web/composer/test_compose_loop_anti_anchor.py -q && \
uv run pytest tests/unit/web/composer/test_advisor_tool.py -q && \
uv run python scripts/cicd/generate_skill_inventory.py --check
```

**Step 5: Commit**

```bash
git add src/elspeth/web/composer/skills/pipeline_composer.md \
  src/elspeth/contracts/composer_llm_audit.py \
  src/elspeth/web/composer/llm_response_parsing.py \
  src/elspeth/web/composer/audit.py \
  src/elspeth/web/composer/tools/sessions.py \
  src/elspeth/web/composer/tools/transforms.py \
  src/elspeth/web/composer/anti_anchor.py \
  src/elspeth/web/composer/tool_batch.py \
  tests/unit/web/composer/test_tool_declarations.py \
  tests/unit/web/composer/test_anti_anchor.py \
  tests/unit/web/composer/test_compose_loop_anti_anchor.py \
  tests/unit/web/composer/test_advisor_tool.py
git commit -m "fix(composer): route narrow edits through splice"
```

**Definition of done:** the composer is explicitly directed to the supported narrow mutation, and an advisor call cannot begin with a budget too small to be useful.

---

### Task 4B: Add the versioned one-shot schema migration required by the target image

This is an operational prerequisite discovered during plan review, not a new runtime migration framework.

**Files:**

- Add: `scripts/migrate_release_0_7_1_aws_ecs_schema.py`
- Modify: `Dockerfile` to copy only that versioned command into the runtime image at a fixed non-secret path
- Modify: `src/elspeth/web/sessions/models.py` to export one immutable PostgreSQL audit-DDL cohort used by both metadata listeners and the migration
- Add: `tests/unit/scripts/test_migrate_release_0_7_1_aws_ecs_schema.py`
- Add: `tests/testcontainer/web/test_release_0_7_1_schema_migration.py`

**Step 1: Pin exact preconditions and an idempotent post-state**

The command accepts the session and landscape schema-owner URLs only through environment variables, requires an explicit `--apply`, never prints URLs, and emits a closed JSON summary. It supports only this exact cohort:

- session pre-state: the release-0.7.0 table shape, no `elspeth_schema_identity`, and none of the six target PostgreSQL audit triggers; post-state: identity epoch 28 plus the six functions/triggers declared in `sessions/models.py`;
- landscape pre-state: release epoch 23 with `run_sources.schema_contract_hash` exactly `VARCHAR(16)` or `VARCHAR(32)` and no identity row; post-state: identity epoch 26, the row-authoritative token FK, artifact-idempotency index, recoverable sink-effect/audit-export tables and triggers, and width exactly 32;
- already-current target state: verify and return `already_applied=true` without DDL;
- the one resumable cross-database partial produced by this command's fixed order (session already target, landscape still recognized pre-state): resume the landscape transaction; reject the reverse or any other mixture.

Any unrecognized mixed/foreign table set, unexpected identity, duplicate/disabled trigger, unexpected hash width, shape mismatch, forged token/run ownership row, or duplicate artifact idempotency key fails before writes with a closed code. The command never truncates, narrows, deletes, or logs row data. PostgreSQL cannot position an added column, while the runtime schema validator treats physical column order as part of the exact contract, so the Landscape transaction creates exact epoch-26 replacements for `operations` and `artifacts`, copies and compares every legacy row projection, rebinds the dependent `calls.operation_id` FK, and drops the two predecessor tables only after those checks. Any failure rolls the entire Landscape transaction back to the byte-stable epoch-23 predecessor.

Open both schema-owner connections, acquire and hold the existing session-level `ELSPETH_SCHEMA_INIT_LOCK_CLASSID` lock on both database-scoped connections in deterministic session-then-landscape order, then preflight both databases. Hold both locks through the session transaction, landscape transaction, and post-verification; release them in reverse order and verify every unlock, using the existing invalidation/closed cleanup semantics on uncertainty. This prevents a preflight-to-DDL race. PostgreSQL cannot atomically span the two databases; after the first commit, failure recovery is explicitly fix-forward and the recognized partial is retryable.

The exact DDL comes from target metadata definitions, not handwritten approximations or SQLAlchemy listener introspection. Refactor the six PostgreSQL function/trigger bodies in `sessions/models.py` into one immutable exported cohort and have both the existing table-scoped `event.listen(..., "after_create", DDL(...))` registrations and the migration consume it. This is required because `metadata.create_all(checkfirst=True)` does not fire those listeners for existing tables. Create the shared identity table with all checks, insert the one checked row, install the shared session cohort, migrate the exact Landscape 23→24→25→26 chain in one transaction, and conditionally widen 16 to 32. Preserve each rebuilt table's exact non-owner ACL; propagate only the DML grants common to every predecessor table onto the new ledger tables, and only common `SELECT` onto each schema-identity table. Do not call `doctor --init-schema`; it deliberately refuses stale existing databases.

**Step 2: Prove data preservation and refusal behavior**

Unit tests pin the closed plan, redacted JSON, precondition classifier, and exact bounded table-replacement vocabulary. PostgreSQL testcontainer coverage builds the genuine epoch-23 pre-target shape rather than deriving a false predecessor from current metadata, inserts representative session/audit/landscape rows, records counts and stable content hashes, runs the command, and asserts:

- every unaffected pre-existing row count/content hash is unchanged, and every rebuilt operation/artifact legacy projection is identical with only the required epoch-26 evidence columns added;
- session and landscape candidate probes are `CURRENT`;
- all six triggers are enabled and reject the protected update/delete operations;
- 16 widens to 32, while an already-32 column is a no-op;
- a second run is byte-stable and reports `already_applied=true`;
- an unrecognized partial, unexpected-width, foreign, or mixed state fails before any DDL;
- forged token ownership and duplicate artifact idempotency records fail before either database changes;
- the recognized session-new/landscape-old partial resumes, while the reverse partial fails closed;
- an injected Landscape replacement failure rolls the complete Landscape transaction back and the exact recognized partial resumes cleanly;
- an owner-run migration preserves runtime-role DML access to rebuilt/new tables and SELECT-only access to schema identity, while the runtime role remains unable to execute migration DDL;
- runtime-role credentials cannot execute any migration DDL.

**Step 3: Commit independently**

Run the unit and PostgreSQL testcontainer files once RED after the tests are written and once GREEN after the command/DDL authority is implemented:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider -m 'testcontainer or not testcontainer' \
  tests/unit/scripts/test_migrate_release_0_7_1_aws_ecs_schema.py \
  tests/testcontainer/web/test_release_0_7_1_schema_migration.py
```

```bash
git add Dockerfile \
  scripts/migrate_release_0_7_1_aws_ecs_schema.py \
  src/elspeth/web/sessions/models.py \
  tests/unit/scripts/test_migrate_release_0_7_1_aws_ecs_schema.py \
  tests/testcontainer/web/test_release_0_7_1_schema_migration.py
git commit -m "feat(ops): migrate release 0.7.1 database identities"
```

**Definition of done:** the target's exact schema delta has a versioned, tested, fail-closed, data-preserving one-shot path; no raw ad-hoc `python -c` or unconditional `ALTER` is needed in AWS.

---

### Task 5: Review the whole implementation and perform one verification boundary

**Files:** Review the complete Tasks 1-4B diff. No production file is introduced solely for review.

**Step 1: Perform the one whole-implementation review**

Review the final diff with four lenses:

- reality: declarations, real paths, generated inventory, redaction manifest/snapshot;
- architecture: single profile-lowering authority, authoritative review reconciliation, one-replacement splice;
- quality: stale/duplicate/unknown review behavior, failure atomicity, private-value leakage;
- systems: replay, deadline boundaries, deployment/rollback fidelity.

Fix every substantive finding before testing. Do not repeat this review after each task.

**Step 2: Run the complete focused package after review corrections**

```bash
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider -m 'testcontainer or not testcontainer' \
  tests/integration/web/composer/test_profile_splice_flow.py \
  tests/testcontainer/web/test_composer_splice_concurrency.py && \
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider \
  tests/unit/web/composer/test_splice_transform_tool.py \
  tests/unit/web/composer/test_authoring_reconciliation.py \
  tests/unit/web/composer/test_profile_policy_validation_adapter.py \
  tests/unit/web/sessions/test_schemas.py && \
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider \
  tests/integration/web/test_bedrock_guardrail_policy.py::test_web_runtime_lowering_and_control_modes_use_generic_policy \
  tests/integration/web/test_bedrock_guardrail_policy.py::test_private_bindings_never_enter_authored_prompt_snapshot_or_policy_evidence \
  tests/integration/web/test_bedrock_guardrail_policy.py::test_profiled_prompt_shield_contract_uses_executable_binding_without_mutating_authored_state \
  tests/integration/web/composer/guided/test_wire_dispatch.py::test_confirm_wiring_lowers_operator_profile_only_for_validation \
  tests/unit/web/sessions/test_interpretation_events_service.py::test_resolve_profiled_llm_review_revalidates_lowered_contract \
  tests/unit/composer_mcp/test_server.py::TestDispatchTool::test_dispatch_constructs_explicit_trained_operator_policy \
  tests/unit/web/composer/test_tools.py::test_apply_pipeline_recipe_reports_unavailable_profile_alias \
  tests/unit/web/composer/test_tools.py::test_apply_pipeline_recipe_reports_profile_unavailable_when_required_profile_is_omitted \
  tests/unit/web/composer/test_tools.py::test_apply_pipeline_recipe_preserves_disabled_llm_plugin_reason \
  tests/unit/web/composer/test_prompts.py::TestBuildContextString::test_includes_validation_summary \
  tests/unit/web/composer/test_dispatch_arms_characterization.py::test_arm_discovery_cache_hit_records_success_with_cache_hit_flag \
  tests/unit/web/composer/test_service.py::TestComposerSingleToolCall::test_explicit_approve_mutating_tool_creates_pending_proposal_without_state_mutation \
  tests/unit/web/composer/test_compose_loop_interpretation_review_dispatch.py::test_fresh_session_set_pipeline_then_request_interpretation_review_persists_pending_event \
  tests/unit/web/composer/test_request_interpretation_review_tool.py::test_02b_opted_out_session_does_not_return_pending_payload \
  tests/unit/web/composer/test_request_interpretation_review_tool.py::test_dedup_second_pending_restage_is_idempotent && \
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider \
  tests/unit/web/composer/test_adequacy_guard.py \
  tests/unit/web/composer/test_redact_tool_call_arguments.py \
  tests/unit/web/composer/test_redact_tool_call_response.py \
  tests/unit/web/composer/test_redaction_completeness_property.py \
  tests/unit/web/composer/test_tool_redaction_dataclass.py \
  tests/unit/web/composer/test_tool_redaction_policy.py \
  tests/unit/web/composer/test_tool_declarations.py && \
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider \
  tests/unit/web/composer/test_anti_anchor.py \
  tests/unit/web/composer/test_compose_loop_anti_anchor.py \
  tests/unit/web/composer/test_advisor_tool.py && \
env -u VIRTUAL_ENV uv run --frozen pytest -n0 -q -p no:cacheprovider -m 'testcontainer or not testcontainer' \
  tests/unit/scripts/test_migrate_release_0_7_1_aws_ecs_schema.py \
  tests/testcontainer/web/test_release_0_7_1_schema_migration.py && \
env -u VIRTUAL_ENV uv run --frozen python scripts/cicd/generate_skill_inventory.py --check && \
npm --prefix src/elspeth/web/frontend ci && \
npm --prefix src/elspeth/web/frontend run typecheck && \
mapfile -t PY_CHANGED < <(
  git diff --name-only --diff-filter=ACM \
    "$(git merge-base release/0.7.1 HEAD)" -- '*.py'
) && \
env -u VIRTUAL_ENV uv run --frozen ruff check "${PY_CHANGED[@]}" && \
env -u VIRTUAL_ENV uv run --frozen ruff format --check "${PY_CHANGED[@]}" && \
git diff --check
```

Run the redaction snapshot writer twice during Task 3; `test_adequacy_guard.py` verifies the committed snapshot matches the manifest. Do not run broad unrelated directories through Ruff.

**Step 3: Prove this plan introduced no additional trust-tier fingerprint drift**

The branch was rebased onto a release tip whose own raw trust-tier findings no longer match the committed baseline. Do not rotate that baseline. Export the exact fetched release tree, scan it and the final worktree with the same installed linter and one empty allowlist, and write the release-relative comparison to `/home/john/.local/state/elspeth/composer-splice-hardening/fingerprint-release-comparison-final.json`:

```bash
RELEASE_TREE="$(mktemp -d)" && \
EMPTY_ALLOWLIST="$(mktemp -d)" && \
git archive origin/release/0.7.1 | tar -xf - -C "$RELEASE_TREE" && \
RELEASE_ROOT="$RELEASE_TREE/src/elspeth" \
BRANCH_ROOT="$PWD/src/elspeth" \
EMPTY_ALLOWLIST="$EMPTY_ALLOWLIST" \
ELSPETH_LINTS_SRC="$PWD/elspeth-lints/src" \
env -u VIRTUAL_ENV uv run --frozen python - <<'PY' \
  > /home/john/.local/state/elspeth/composer-splice-hardening/fingerprint-release-comparison-final.json
import json
import os
import subprocess
import sys

env = {**os.environ, "PYTHONPATH": os.environ["ELSPETH_LINTS_SRC"]}
def capture(root: str) -> set[tuple[str, str]]:
    result = subprocess.run(
        [sys.executable, "-m", "elspeth_lints.core.cli", "check",
         "--rules", "trust_tier.tier_model", "--format", "json",
         "--root", root, "--allowlist-dir", os.environ["EMPTY_ALLOWLIST"]],
        capture_output=True, text=True, check=False, env=env,
    )
    if result.returncode not in (0, 1):
        raise SystemExit(result.stderr or result.stdout)
    return {(str(row["file_path"]), str(row["fingerprint"])) for row in json.loads(result.stdout)}

release = capture(os.environ["RELEASE_ROOT"])
branch = capture(os.environ["BRANCH_ROOT"])
added = sorted(branch - release)
removed = sorted(release - branch)
payload = {
    "schema": "elspeth.composer-splice.release-fingerprint-comparison.v1",
    "release_sha": subprocess.run(["git", "rev-parse", "origin/release/0.7.1"], capture_output=True, text=True, check=True).stdout.strip(),
    "branch_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip(),
    "release_count": len(release),
    "branch_count": len(branch),
    "added": [{"file_path": path, "fingerprint": fp} for path, fp in added],
    "removed": [{"file_path": path, "fingerprint": fp} for path, fp in removed],
}
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if not added and not removed else 1)
PY
STATUS=$?
rm -rf "$RELEASE_TREE" "$EMPTY_ALLOWLIST"
exit "$STATUS"
```

Require `added=[]`, `removed=[]`, and equal counts. A difference is plan-created drift and must be resolved before proceeding. The known baseline self-consistency failure is acceptable only when this exact release-relative proof passes; do not regenerate `tests/unit/elspeth_lints/fixtures/fingerprint_baseline.json` or rotate signed entries.

**Step 4: Commit review corrections if needed**

```bash
git add <only substantive review-correction files>
git commit -m "fix(composer): address splice review"
```

Skip the commit when review produces no changes.

**Step 5: Pause for the operator's one full suite**

Give the operator this exact parallel command:

```bash
env -u VIRTUAL_ENV uv run --frozen pytest -n 12
```

Record the exact command, exit code, pass/fail counts, and failure node IDs. The canonical comparison in Step 3, not a verbal exception, is what distinguishes the known fingerprint failure. Investigate every other failure. Do not build or deploy before the focused package and operator suite are accepted.

**Definition of done:** no substantive whole-diff finding remains; the complete focused package passes; the operator records the single parallel full-suite result; the code being deployed is the reviewed code.

---

### Task 6: Stage the timeout, migrate safely, deploy the reviewed image, and verify live

**Repository files:** none after Tasks 1-4B.

**Authoritative acceptance state:**

- State root: /home/john/.local/state/elspeth/aws-ecs/plan12-5db06e8cb-attempt18/live.
- Terraform root: terraform/scenario-a, workspace acceptance-a, with protected/scenario-a.tfvars and protected/scenario-a.binding.json.
- Public origin: https://elspeth-aws.foundryside.dev.
- AWS profile and region: elspeth-acceptance and ap-southeast-1.

Scenario B is not deployed. Never plan or apply it. At plan-correction time the recovered live state was one healthy web:24 deployment at SHA 4c271bf0f, timeout 120, both schemas CURRENT, and ALB idle timeout 300. Treat those as volatile evidence and re-query before every mutation.

**Step 1: Re-establish one healthy owner state and pin the deployment procedure**

Authenticate and capture the caller, service, task definition, image digest/SHA, target group, listener/rule actions, auth provider, and current ECR digest existence. If credentials are expired, pause for browser login. If another ECS deployment is IN_PROGRESS, do not overlap it; poll to a terminal state and recover the last known healthy pre-schema task only if no migration task ran.

Acquire an owner-scoped deployment lease as a conditional-create S3 object in the already-bound encrypted backend bucket, containing task ID, actor, start time, expiry, worktree SHA, and expected live service fingerprint. If an unexpired object already exists for this exact task/owner/SHA, verify the fingerprint and absence of foreign events, then resume and renew only with an ETag-conditional write. If this task's lease expired after a crash, perform the same full state/event audit before conditional renewal. A foreign unexpired lease blocks mutation. A foreign expired lease may be archived and conditionally removed only after two separated service/CloudTrail observations show no active deployment or foreign mutation; otherwise stop for ownership resolution. Verify lease content/ETag before conditional release. Immediately before and after every Terraform apply, RunTask, UpdateService, listener-rule change, snapshot, and schema action, re-read the lease, service fingerprint, and CloudTrail events since the prior checkpoint. Abort on a foreign actor/event or unexpected state transition. The lease is coordination evidence, not permission to ignore live drift.

Require exactly one PRIMARY/COMPLETED deployment, desired=running=1, pending=0, one healthy target, /api/health 200, /api/ready 200, and both schema checks CURRENT. Capture CloudTrail RegisterTaskDefinition, UpdateService, and RunTask events so an unowned migration cannot be mistaken for a failed web rollout.

Pin the current external runbook from /home/john/elspeth/.worktrees/aws-ecs-program/docs/runbooks/aws-ecs-deployment.md: record its source commit and SHA-256, copy that exact version into the owner-only live state, and use its verify_tf_binding and sanitized-evidence workflow. Do not rely on the stale runbook bundled with the original package.

**Step 2: Reconstruct state-aligned Scenario A Terraform before changing behavior**

The five-file package diff is applied-state reality, including the live single RDS instance; clean git HEAD describes the obsolete Aurora shape and is destructive. Leave the dirty package byte-for-byte untouched: record its status and per-file SHA-256, create a separate external Terraform worktree from binding `repository_commit` 52e52457633b10c8ecda5c348a2b38385a06f7e1, and replay the reviewed five-file applied-state diff into that detached worktree with `apply_patch`. Recheck the original dirty package hashes afterward. Commit only in the detached external worktree on a dedicated branch; do not revert/clean/commit the original package and do not alter the ELSPETH repository.

Add create-only immutable resources with empty-map/default rendering so the baseline remains byte-identical:

- An append-only timeout-web map keyed by an immutable operation ID; each entry creates a separately addressed task definition from the captured live image/SHA with only the web timeout changed.
- An append-only candidate map keyed by full reviewed SHA plus digest identity; each entry creates separately addressed web, doctor, and schema-migration task definitions with candidate-specific image/SHA/environment. Do not reuse the shared `var.candidate_image`, `var.candidate_sha`, helper task definitions, or observability locals, so payload, local-auth, database-bootstrap, dashboards, and alarms remain unchanged.
- A schema-migration task container that invokes the checked-in Task 4B command with schema-owner secrets and has no automatic `terraform_data` provisioner.
- A maintenance-mode variable that renders the existing `aws_lb_listener_rule.traffic` action exactly as forward by default and as fixed maintenance only when explicitly enabled. The default listener itself remains unchanged.

Never remove an entry during this task. Terraform plans for task definitions must contain create-only actions at new addresses; any delete or create/delete replacement aborts. The captured 120-second and staged 270-second task definitions remain registered through the pre-schema recovery window.

Update the owner-only Scenario A binding to the exact clean external-worktree commit, lock, vars, backend, and `acceptance-a` workspace; explicitly select that workspace before running verify_tf_binding. First run and save a full baseline plan and JSON; it must contain zero non-noop changes. If not, reconcile source and state before continuing. Never use -target, refresh suppression, or an apply as reconciliation.

This binding check is per phase, not one-time. After every timeout-map append, candidate-map append (including fix-forward), maintenance-mode toggle, or other detached Terraform/protected-tfvars edit: commit the detached worktree, update the owner-only binding to that exact repository commit plus current lock/vars/backend/workspace hashes, select `acceptance-a`, and rerun `verify_tf_binding` before planning. Recheck the same binding immediately before applying the saved plan. Never plan/apply from uncommitted or binding-stale configuration.

Every later plan JSON is machine-gated. Reject every delete or create/delete replacement, every non-noop address outside that phase's allowlist, `terraform_data.database_bootstrap`, and every protected resource type/address covering RDS/Aurora, ALB/listeners/rules/target groups, EFS, Cognito, Secrets Manager, session/audit storage, and Bedrock policy resources. The sole protected-resource exception is an exact in-place `update` of `module.scenario.aws_lb_listener_rule.traffic` during the explicit maintenance-on and maintenance-off phases; verify its before/after action JSON and reject any other field change. Save binary plans and JSON and apply only the exact gated saved plan.

The baseline allowlist is empty. The timeout phase allows only one new immutable timeout-web task-definition address with actions exactly `["create"]`. Candidate preparation allows only the three new immutable web, doctor, and schema-migration addresses for that candidate, each create-only. Maintenance phases allow only the exact listener-rule in-place update. For each plan, select every resource change whose actions are not exactly no-op and assert its address/action pair is in the closed phase allowlist; separately enforce the protected-resource exception above. A destructive or out-of-allowlist plan is the authorized pause boundary.

**Step 3: Stage and prove the 270-second timeout on the current healthy image**

Append one timeout-web entry for 270 while retaining the exact live image digest and SHA. The timeout plan must contain exactly one new immutable task-definition address with actions `["create"]`; no existing resource changes. Rendered container definitions must differ from the live task only in timeout, family/operation identity, and expected revision metadata. ALB is verify-only and must remain 300.

Apply the saved plan to register the task definition. The ECS service intentionally ignores task_definition, so explicitly call aws ecs update-service with the registered ARN. Poll every 10 seconds to a bounded deadline. Success requires one exact PRIMARY/COMPLETED deployment, desired=running=1, pending=0, failedTasks=0, one RUNNING task, and the exact healthy ENI/port target. Fail immediately on FAILED, failedTasks greater than zero, SERVICE_DEPLOYMENT_FAILED, an essential-container exit, or a stopped reason.

Verify /api/health, /api/ready, and /api/system/status.composer_timeout_seconds == 270 without disabling TLS verification. Use the same HTTPS origin in Playwright and assert the browser guard is exactly 295 while the live ALB attribute is exactly 300. These are pinned acceptance ceilings with zero headroom. Before schema migration, recovery may explicitly select the captured healthy 120-second task definition after re-verifying its digest still exists.

**Step 4: Build or reuse one immutable reviewed candidate and register preparation tasks**

Use tag composer-splice-<first-12-candidate-SHA>. Query ECR before building:

- If absent, build once for linux/amd64, label org.opencontainers.image.revision with the exact reviewed SHA, push once, and resolve the immutable digest.
- If present, do not overwrite it; pull by digest and require its revision label to equal the exact reviewed SHA before reuse.
- If present with a different revision, fail closed and use a collision-free full-SHA tag rather than overwriting.

Immediately re-verify every digest relied upon because lifecycle rules may expire old acceptance tags. Pass only repository@sha256 references to Terraform.

Append one candidate-map entry. The candidate-preparation plan may create only its new immutable web, doctor, and schema-migration task definitions; it may not replace or delete anything. Scenario A rollback task definitions are absent and must not be invented. Apply the gated saved plan to register definitions only. Assert the service still runs the healthy timeout-staged old image. Any later code-repair SHA appends another entry and retains all earlier recovery definitions.

**Step 5: Preserve, drain, migrate, and prove the target schema before traffic**

Use deterministic snapshot ID `composer-splice-pre-ddl-<first-12-reviewed-SHA>`. If absent, create it and wait until `available`; if present, reuse it only when it is `available`, belongs to the exact live DB instance, and predates all migration events for this task. Otherwise fail closed. Record pre-migration table counts and closed structural/schema facts without row contents.

Set maintenance mode in the detached Terraform configuration, plan an exact in-place update of only `module.scenario.aws_lb_listener_rule.traffic` from forward to fixed maintenance, machine-gate/apply it, and verify live action JSON. Then explicitly scale the ECS service to zero; desired count remains intentionally ignored by Terraform, so later refreshes do not create plan drift. Require no running web task before DDL.

Run the Terraform-owned schema-migration Fargate task with the reviewed candidate digest. Require task exit 0 and its sanitized closed JSON result to report `session_state=current`, `landscape_state=current`, and `already_applied=false` (or the exact idempotent-current result on a proven retry). Width 32 is an expected no-op on the current live database; width 16 may only widen. Then run the candidate doctor task and require exit 0, every check ok, session epoch 28, Landscape epoch 26, six enabled session audit triggers, the epoch-26 Landscape trigger/index/constraint manifest, and Landscape width 32.

From the first migration DDL onward, the old task definition is forbidden because it cannot validate the new identity table and epochs. Failure handling is maintenance response plus service zero plus bounded sanitized logs plus fix-forward and another candidate doctor. A configuration/task-definition-only repair may reuse the same reviewed digest. Any code repair changes the SHA/digest and must return to Task 5 for whole-diff review, focused/fingerprint verification, and new operator full-suite evidence before cutover; keep the Terraform maintenance variable enabled and desired count zero meanwhile, then append a new create-only candidate-map entry. This is the pre-authorized registration path while maintenance is active and does not require a false zero-drift baseline. Never narrow, delete, replace, or automatically restore either database. The RDS snapshot is emergency preservation evidence; restoration is a separate destructive operator decision.

**Step 6: Cut over explicitly and diagnose transparently**

Call update-service with the exact candidate task definition and desired count 1. Use the bounded poll above; ecs wait services-stable alone is insufficient. While the listener rule still serves maintenance, require the exact candidate task to be running, its target-group health check to be healthy, and bounded sanitized ECS Exec requests to localhost to pass `/api/health`, `/api/ready`, and system status 270; also require task-definition image, SHA, and digest to match. Then set Terraform maintenance mode false, machine-gate/apply the exact in-place listener-rule update back to the captured forward action, verify live action JSON, and repeat both probes and system status through the HTTPS public origin. This avoids the circular requirement to pass public probes while the rule is deliberately fixed-response.

On failure, buffer describe-tasks and CloudWatch diagnostics and pass them through python -m elspeth.web.aws_ecs_acceptance sanitize-evidence with kind web-log or doctor-log before persistence or display. Record ECS events, stopped reason, essential exit code, and bounded sanitized log events. Do not use opaque waits or raw log pipes.

**Step 7: Establish same-origin authenticated browser evidence**

Detect auth from the deployed task definition. Scenario A currently uses local auth and open registration. Reuse a dedicated owner-held acceptance account from `$LIVE/protected/composer-splice-local-auth.json` (directory mode 0700, file mode 0600); create it through `POST https://elspeth-aws.foundryside.dev/api/auth/register` only if it does not yet exist, because the public API has no account-deletion endpoint. Never create an accumulating one-use account.

Use the `playwright-interactive` persistent browser driver directly rather than `playwright.staging.config.ts`, so bearer state stays in the browser context and no default-mode `.auth/staging-user.json`, trace, video, or screenshot artifact is written. The exact driver flow is:

1. Create an in-memory Playwright request context with base URL `https://elspeth-aws.foundryside.dev`, log in using the owner-only artifact, and retain the bearer only in memory.
2. Open the same origin in Chromium, install `auth_token` into that origin's local storage, reload, and verify authenticated `GET /api/auth/me`.
3. Perform the UI action in Step 8, then use the same in-memory bearer for `GET /api/sessions/{id}/messages?include_tool_rows=true&include_llm_audit=true&limit=500` and the exact preview/session evidence endpoints.
4. Delete only auxiliary auth-smoke sessions through `DELETE /api/sessions/{id}`. Preserve the successful splice session and its audit rows as the required live evidence. Close browser/request contexts, clear local storage, and confirm no trace/video/screenshot/storage-state file was created. Sanitize then delete any failure artifact before leaving the phase.

If live detection instead reports OIDC, use `playwright.oidc.config.ts` only with protected `.oidc.username/password` fields in `$LIVE/protected/operator-inputs.json`, `umask 077`, a 0700 handoff directory, a 0600 `OIDC_BEARER_HANDOFF_FILE`, and trace/video/screenshot disabled. Read that handoff token into the same interactive browser context used for Step 8, install it only in the exact HTTPS origin's local storage, then securely remove the handoff and storage state after the run. Pause only if that artifact is genuinely missing or unusable, or MFA requires a human. Never mix frontend and backend origins and never disable TLS verification.

**Step 8: Repeat the original action and prove declaration separately from selection**

In a fresh authenticated session, create the same web_scrape -> llm shape and send exactly:

    insert a content guard between the source and the LLM

Fetch authenticated session messages with include_tool_rows=true, include_llm_audit=true, and a bounded limit. Prove independently:

Task 4's unit/integration evidence is the integrity proof that `tools_spec_hash` equals the exact transmitted specification after provider transformation. The live run proves the deployed sidecar declares `splice_transform` and carries a non-null hash; it does not reconstruct the full private provider payload from the session API.

- Declaration: the mutation-turn LLM sidecar has a non-null tools_spec_hash and declared_tool_names containing splice_transform.
- Selection: exactly one separate tool audit invocation selects splice_transform, and none selects set_pipeline.
- State: exactly one version increment, canonical node/edge order, valid final preview, and identical replay is a no-op.
- Policy and review: the profile-backed shield guarantees page_text, its recommendation is retired, unrelated accepted reviews survive, and no raw-contract probe warning appears.
- Secrecy and deadline: no private binding or sentinel appears in returned, audit, log, or YAML surfaces; no advisor call starts below five seconds; and a below-floor refusal has no LLM row.
- Timing: backend, browser, and ALB remain exactly 270, 295, and 300 seconds.

Record image digest, task-definition ARN, session ID, elapsed time, declaration names/hash, selected tool sequence, state-version transition, and final validation result in the Filigree completion comment. Do not add a permanent proof-only run sheet.

After every acceptance assertion and tracker comment is durable, recheck CloudTrail/service state once more and release the S3 deployment lease only when its content still matches this task and owner.

**Definition of done:** the reviewed digest is the sole healthy deployment; both target schemas are CURRENT; backend/browser/ALB ceilings are 270/295/300; the original instruction succeeds through one declared-and-selected atomic splice; evidence proves review preservation and private-binding secrecy; the tracker task closes only then.

---
## Explicit non-goals

- A general graph patch language or arbitrary branch splice semantics.
- Automatic insertion across gates, forks, coalesces, queue branches, error routes, or sinks.
- Making persisted audit/private state directly editable by the composer.
- Inferring detailed mutation intent inside the bounded anti-anchor tracker.
- Adding mandatory advisor calls.
- Moving generated acceptance Terraform into the repository.
- Running CI, Wardline, signing, or the full suite after each task.
