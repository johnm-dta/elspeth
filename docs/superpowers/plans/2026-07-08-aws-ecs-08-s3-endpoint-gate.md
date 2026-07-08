# AWS S3 Web-Authorship Endpoint Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Reject `endpoint_url` on web-authored `aws_s3` source/sink configs on every web authoring path while CLI/batch pipelines keep the field.

**Architecture:** A pure policy predicate in `provider_config_policy.py` (mirrors `web_llm_base_url_policy_error`) keys on `plugin == "aws_s3"` + the `endpoint_url` key. **The load-bearing gate is `validate_pipeline()`** in `web/execution/validation.py`, extended to loop `state.sources`/`state.outputs` (today's policy loop only covers `state.nodes` transforms) — this is the choke point every web-authored `CompositionState` converges through before it can run. Verified directly (not inherited from the review): `execution/service.py::execute()` calls `validate_pipeline(...)` unconditionally at line 667 — a sequential statement, not nested under any conditional — immediately before `create_run`, and raises `PipelineValidationError` if invalid (service.py:675-676). Separately, `_state_data_from_composer_state()` (`sessions/routes/_helpers.py:1593`) invokes it via `_runtime_preflight_for_state` (`_helpers.py:1175`, which calls `validate_pipeline` directly) at every state-persist boundary, including composer finalize *and* YAML import/seed (`routes/composer/state.py::import_state_yaml`/`seed_state_for_e2e`, both call `_state_data_from_composer_state` with `runtime_preflight=None`): that argument name is misleading — passing `None` does not skip validation, it tells the helper to compute a fresh `validate_pipeline()` run itself (`_helpers.py:1624-1626: "if runtime is None and authoring.is_valid: runtime = await _runtime_preflight_for_state(...)"`) whenever the cheaper structural "authoring" check passes; if authoring itself fails, the state persists as `is_valid=False` and can never execute either way. `preflight_exception_policy="persist_invalid"` only governs what happens if that computation *raises* — persist the failure diagnostics rather than 500 — it does not skip the call.

Six sites construct `SourceSpec`/`OutputSpec` from web input (verified exhaustively via `grep -rn 'SourceSpec(\|OutputSpec(' src/elspeth/web/`); all six converge on the chokepoint above, and five get an additional Task-3 tool-level rejection as defense-in-depth (immediate LLM-facing feedback instead of a delayed finalize/execute-time failure):
- `set_source` (`composer/tools/sources.py:512`) — gated (Task 3).
- `patch_source_options` (`composer/tools/sources.py:924`, via `dataclasses.replace`) — gated (Task 3).
- `set_output` (`composer/tools/outputs.py:167`) — gated (Task 3).
- `patch_output_options` (`composer/tools/outputs.py:252`, via `dataclasses.replace`) — gated (Task 3).
- `set_pipeline` (`composer/tools/sessions.py:312`/`:446` source, `:652` output) — gated (Task 3 extension, binding decision #2). `apply_pipeline_recipe` (`sessions.py:713`) delegates unconditionally to `_execute_set_pipeline` (`sessions.py:784`) and inherits the gate transitively — no separate construction site, no separate check needed.
- `set_source_from_blob` (`composer/tools/sources.py:589`) — **not** tool-gated: chokepoint-only. Narrow/speculative surface (`_resolve_source_blob` force-injects `path`/`blob_ref`/`mode`, which the future aws_s3 config model's likely `extra: forbid` — mirroring `azure_blob_source.py:70,98` — will probably reject anyway) and not independently testable pre-06/07.
- YAML-paste/import (`yaml_importer.py:143`/`:247`, plus `_state_with_imported_source_blobs` at `routes/composer/state.py:226`, which copies an already-parsed source's options forward untouched) — **not** tool-gated (a parse/route path, not an LLM tool call): chokepoint-only, per the persist-time mechanism verified above.

**Tech Stack:** Python, pydantic, pytest — no new dependencies.

**Depends on:** none for build/test — Tasks 1-3 use synthetic `SourceSpec`/`OutputSpec` and a mocked catalog, so this plan builds and passes standalone before the real `aws_s3` plugin exists. **Sequencing matches the overview's wave plan (`00-overview.md`, Execution order) — not the reverse:** Tasks 1-2 are **Wave 1**, parallel with plan 06, and load-bearing — the policy predicate and the `validate_pipeline()` choke-point gate must be in the tree before, or in the same merge as, the first web-reachable `aws_s3` registration from 06/07 (auto-discovery makes registration = web-catalog-reachable, per plan 06's own `Depends on:`). Do not defer Tasks 1-2 until after 06/07 land — that would put the exact web-authorable exfiltration/SSRF surface the design review rated Critical into a mergeable tree with no gate. Task 3 (composer-tool defense-in-depth) is **Wave 2**, scheduled after 06/07 in the program plan as a scheduling choice, not a code or file dependency: Task 3 also builds against the mocked catalog (see its Interfaces note — `_mock_catalog()` accepts `plugin="aws_s3"` with no real plugin registered), and its only test file, `tests/unit/web/composer/test_tools.py`, is untouched by plans 06/07 (verified: neither plan references `test_tools.py` or `test_validation.py`). Real end-to-end CLI acceptance of an `aws_s3` pipeline with `endpoint_url` belongs to **plan 06's** own suite once the config model exists — **named cross-plan obligation, owned by plan 06 (Task 3):** plan 06 adds `test_cli_aws_s3_endpoint_url_accepted` (a CLI/batch-loaded YAML with an `aws_s3` source carrying `endpoint_url`, run through `load_settings_from_yaml_string` **and** `instantiate_plugins_from_config` — the latter is the real acceptance gate, since `SourceSettings.options` is an untyped `dict[str, Any]` (`config.py:1061`) and only plugin instantiation runs `AwsS3SourceConfig`'s own field validation; `load_settings_from_yaml_string` alone would accept `endpoint_url` regardless of the plugin). The sink-side equivalent (an `aws_s3` output's `endpoint_url`) is owned by plan 07 (Task 3, `test_cli_aws_s3_sink_endpoint_url_accepted`), mirroring plan 06's source-side test. This plan proves only the negative half: the web gate is unreachable from `elspeth.core.config`, the settings loader the CLI `run`/`resume` commands use (Task 2's import-boundary test) — it does not and cannot prove CLI retention before the aws_s3 model exists.

**Global Constraints:** "Web-authored `aws_s3` configuration must not set `endpoint_url`... The gate lives at the same seam as `web_llm_base_url_policy_error`: `web/execution/validation.py`, backed by `provider_config_policy`. `endpoint_url` remains available to CLI/batch-authored pipelines."

### Task 1: `web_aws_s3_endpoint_url_policy_error` predicate

**Files:**
- Modify: `src/elspeth/web/provider_config_policy.py`
- Test: `tests/unit/web/execution/test_validation.py` (new class, mirrors `TestWebLlmBaseUrlPolicyHelper` at line 750)

**Interfaces:** Produces `web_aws_s3_endpoint_url_policy_error(plugin: str | None, options: Mapping[str, Any]) -> str | None` and `AWS_S3_ENDPOINT_URL_POLICY_ERROR: Final[str]`.

- [ ] Write `TestWebAwsS3EndpointUrlPolicyHelper` with `test_non_aws_s3_plugin_is_ignored` (`web_aws_s3_endpoint_url_policy_error("csv", {"endpoint_url": "http://evil"}) is None`), `test_unset_endpoint_url_is_allowed` (`web_aws_s3_endpoint_url_policy_error("aws_s3", {"bucket": "b", "key": "k"}) is None`), `test_endpoint_url_is_blocked` (`web_aws_s3_endpoint_url_policy_error("aws_s3", {"endpoint_url": "http://127.0.0.1:4566"}) is not None`), `test_none_plugin_is_ignored` (`web_aws_s3_endpoint_url_policy_error(None, {"endpoint_url": "x"}) is None`), `test_non_string_endpoint_url_is_blocked` (`web_aws_s3_endpoint_url_policy_error("aws_s3", {"endpoint_url": 123}) is not None` — the predicate is intentionally type-naive: it blocks on mere presence, unlike the base_url helper's normalization branch).
- [ ] Run `pytest tests/unit/web/execution/test_validation.py::TestWebAwsS3EndpointUrlPolicyHelper -x` — fails with `ImportError: cannot import name 'web_aws_s3_endpoint_url_policy_error'`.
- [ ] Implement in `provider_config_policy.py`, appended after `web_llm_base_url_policy_error` (the function's `return LLM_BASE_URL_POLICY_ERROR` ends the file at line 144 — append at EOF, there is no line 145 yet):
  ```python
  AWS_S3_ENDPOINT_URL_POLICY_ERROR: Final[str] = (
      "Web-authored aws_s3 source/sink options may not set endpoint_url. AWS "
      "credentials resolve from the server's ambient default credential chain "
      "(the ECS task role), so an author-chosen endpoint_url would redirect "
      "signed requests to it — pipeline-payload exfiltration / SSRF. Omit it; "
      "CLI/batch pipelines (LocalStack, tests) keep the field because that "
      "single-machine threat model does not hold for a hosted server."
  )


  def web_aws_s3_endpoint_url_policy_error(plugin: str | None, options: Mapping[str, Any]) -> str | None:
      """Reject web-authored aws_s3 source/sink configs that set endpoint_url."""
      if plugin != "aws_s3":
          return None
      if options.get("endpoint_url") is None:
          return None
      return AWS_S3_ENDPOINT_URL_POLICY_ERROR
  ```
- [ ] Run the same command — PASS.
- [ ] `git add src/elspeth/web/provider_config_policy.py tests/unit/web/execution/test_validation.py && git commit -m "feat(web): add aws_s3 endpoint_url web-authorship policy predicate"`

**Skipped (low, by design):** parameterizing `AWS_S3_ENDPOINT_URL_POLICY_ERROR` by `component_type` (source vs. sink) to name the exact vector (server issues signed requests to an author-chosen host vs. server PUTs output to one) — the message already says "source/sink options" and "SSRF" generically, and splitting it would ripple the predicate's `(plugin, options)` signature across all 6 call sites (Task 2 ×2, Task 3 ×5) for marginal debugging value. Kept as one shared string.

**Coordination note for plans 06/07 (low):** the predicate keys on a literal top-level `options["endpoint_url"]` with no casing/alias/nested-passthrough handling. This is sound only if the real `aws_s3` config model exposes `endpoint_url` as a canonical top-level field with no alias and `extra: forbid` (mirroring `azure_blob_source.py:70,98`) and no nested boto-client-config dict that could carry an equivalent option. Plans 06/07 should confirm the model matches this shape; if it doesn't, this predicate needs a follow-up case.

### Task 2: Wire into `validate_pipeline()`

**Files:**
- Modify: `src/elspeth/web/execution/schemas.py:43` (Literal), `:65` (constant, after `CHECK_LLM_BASE_URL_POLICY`), `:86` (tuple, after `CHECK_LLM_BASE_URL_POLICY`) — add `"aws_s3_endpoint_url_policy"` / `CHECK_AWS_S3_ENDPOINT_URL_POLICY`.
- Modify: `src/elspeth/web/execution/validation.py:116-120` (import `web_aws_s3_endpoint_url_policy_error`), `:133` (alias `_CHECK_AWS_S3_ENDPOINT_URL_POLICY`), insert new loop after line 1620 (immediately before the `# Step 3: Settings loading` comment at 1622).
- Modify: `tests/unit/web/execution/test_validation.py:53-60` (`_make_source`: add `plugin: str = "csv"` param), `:99-117` (`_make_state`: add `source_plugin: str = "csv"` param, threaded into its `_make_source(...)` calls), `:723` (add `"aws_s3_endpoint_url_policy"` to the `later_check` tuple in `test_web_scrape_failure_skips_later_managed_identity_and_llm_retry_checks`), `:2020` (`assert len(result.checks) == 16` → `17`; a new passing check is emitted on every happy path).
- Test: same file, new `TestValidatePipelineAwsS3EndpointUrlPolicy` class (mirrors `TestValidatePipelineLlmBaseUrlPolicy` at line 787), plus one architecture-boundary test.

**Interfaces:** Consumes `state.sources: Mapping[str, SourceSpec]` and `state.outputs: tuple[OutputSpec, ...]` (`composer/state.py:1990/1986`) — `state.nodes` never carries `node_type in ("source","sink")`, only `"transform"/"gate"/"aggregation"/"coalesce"` (`composer/state.py:150`), so this new loop is the only way source/sink options reach a policy check. `_ALL_CHECKS` (`validation.py:201`) is `list(VALIDATION_BLOCKING_CHECK_NAMES)` — declaring the new name in the tuple (schemas.py) is sufficient; no separate `_ALL_CHECKS` edit needed, and the order-sync test (`test_validation.py:3583`, `assert tuple(_ALL_CHECKS) == VALIDATION_BLOCKING_CHECK_NAMES`) and the physical-emission-order test (`:3624`) both pass automatically once the check is declared in the tuple and appended in the matching physical position — verified by reading both tests.

- [ ] Extend the test helpers first: add `plugin: str = "csv"` to `_make_source`, and `source_plugin: str = "csv"` to `_make_state` threaded into its internal `_make_source(...)` calls (`_make_output` already accepts `plugin`; only the source side is missing it — `_make_state` otherwise has no way to express a non-csv source, since it only ever builds `CompositionState(source=..., ...)` via `_make_source`). Then write the two new test cases: `_make_state(source_options={"endpoint_url": "http://evil"}, source_plugin="aws_s3", outputs=(_make_output(name="results"),))` and `_make_state(outputs=(_make_output(plugin="aws_s3", options={"endpoint_url": "http://evil"}, name="results"),))`, each asserting `result.is_valid is False`, `_check(result, "aws_s3_endpoint_url_policy").passed is False`, `result.errors[0].error_code == "aws_s3_endpoint_url_not_allowed"` — built like `test_loopback_base_url_rejected_before_yaml_generation` (mock `load_settings_from_yaml_string` to short-circuit after).
- [ ] Also add `test_core_config_loader_has_no_web_dependency`: `assert "elspeth.web" not in inspect.getsource(elspeth.core.config)`. This is a standing architectural regression guard, not a red/green step for this feature: it is already true on a clean checkout today (verified — `elspeth.core.config`, the settings loader the CLI `run`/`resume` commands use via `load_settings`, has zero reference to `elspeth.web`) and stays true because nothing in Tasks 1-3 touches `core/config.py`. It proves the negative half only — the new gate is structurally unreachable from the CLI's settings loader — not that CLI/batch pipelines retain `endpoint_url`; see this plan's `Depends on:` for the named obligation that closes the positive half in plans 06/07.
- [ ] Run `pytest tests/unit/web/execution/test_validation.py -k AwsS3EndpointUrlPolicy -x` — fails (unknown check name / no rejection). `test_core_config_loader_has_no_web_dependency` is not part of this red step — it passes immediately since it asserts a pre-existing invariant; run it once standalone to confirm before moving on.
- [ ] Add the check-name plumbing (schemas.py) and, in `validation.py` after line 1620:
  ```python
  for source_name, source in state.sources.items():
      endpoint_policy_error = web_aws_s3_endpoint_url_policy_error(source.plugin, source.options)
      if endpoint_policy_error is not None:
          component_id = "source" if source_name == "source" else f"source:{source_name}"
          checks.append(ValidationCheck(name=_CHECK_AWS_S3_ENDPOINT_URL_POLICY, passed=False,
              detail=f"Source '{source_name}' sets disallowed aws_s3 endpoint_url", affected_nodes=(component_id,), outcome_code=None))
          _append_skipped_checks(checks, _CHECK_AWS_S3_ENDPOINT_URL_POLICY)
          return ValidationResult(is_valid=False, checks=checks,
              errors=[ValidationError(component_id=component_id, component_type="source", message=endpoint_policy_error,
                  suggestion="Remove endpoint_url; aws_s3 uses the AWS default credential chain.", error_code="aws_s3_endpoint_url_not_allowed")],
              readiness=_blocked_readiness(code=_CHECK_AWS_S3_ENDPOINT_URL_POLICY, detail=f"source '{source_name}' sets disallowed aws_s3 endpoint_url",
                  component_id=component_id, component_type="source"),
              semantic_contracts=serialize_semantic_contracts(semantic_contracts))
  ```
  Mirror this exactly for `for output in state.outputs or ():` using `output.plugin`, `output.options`, `component_id=output.name`, `component_type="sink"` (matches the existing path-allowlist sink loop at `validation.py:918`). Close both with one `checks.append(ValidationCheck(name=_CHECK_AWS_S3_ENDPOINT_URL_POLICY, passed=True, detail="No web-authored aws_s3 endpoint_url override", affected_nodes=(), outcome_code=None))`.
- [ ] Update the `== 16` assertion to `17`. Add `"aws_s3_endpoint_url_policy"` to the `later_check` tuple in `test_web_scrape_failure_skips_later_managed_identity_and_llm_retry_checks` (test_validation.py:723) so the existing skipped-after-failure regression test extends to the new check.
- [ ] Run the same pytest command plus `pytest tests/unit/web/execution/test_validation.py -x` (full file) — PASS.
- [ ] `git add src/elspeth/web/execution/schemas.py src/elspeth/web/execution/validation.py tests/unit/web/execution/test_validation.py && git commit -m "feat(web): enforce aws_s3 endpoint_url gate in pipeline validation"`

### Task 3: Wire into composer source/output mutation tools

**Files:**
- Modify: `src/elspeth/web/composer/tools/sources.py:490-491` (`_execute_set_source`, after `review_metadata_error`), `:898-899` (`_execute_patch_source_options`, after `new_options = _apply_merge_patch(...)`).
- Modify: `src/elspeth/web/composer/tools/outputs.py:141-142` (`_execute_set_output`, after `sink_options = validated.options`), `:224-225` (`_execute_patch_output_options`, after `new_options = _apply_merge_patch(...)`).
- Modify: `src/elspeth/web/composer/tools/sessions.py:295-304` (`_execute_set_pipeline`, `sources` mapping branch, after its `credential_error` check), `:346-355` (same function, legacy single-`source` branch, after its `credential_error` check — before the blob-resolution branches that follow), `:564-573` (same function, output loop, after its `credential_error` check).
- Test: `tests/unit/web/composer/test_tools.py`, classes `TestSetSource` (~455), `TestSetOutput` (~1960), `TestPatchSourceOptions` (~6094), `TestPatchOutputOptions` (~6425), `TestSetPipeline` (~7450).

**Interfaces:** Consumes `_failure_result(state, error_msg) -> ToolResult` (`_common.py:868`) and Task 1's predicate. **Scope note (binding decision #2):** Task 2's `validate_pipeline()` is the load-bearing gate — every `SourceSpec`/`OutputSpec` construction site converges on it before a pipeline can persist-as-valid or execute (full six-site enumeration and each site's coverage disposition is in this plan's Architecture section). The five tool handlers gated in this task (four below, plus `set_pipeline` — which transitively covers `apply_pipeline_recipe` by unconditional delegation at `sessions.py:784`) are defense-in-depth: immediate LLM-facing rejection instead of a delayed finalize/execute-time failure. `set_source_from_blob` and the YAML-import path are deliberately **not** given a tool-level gate here — they rely solely on the Task 2 chokepoint, which is proven (not assumed) to run for both, per the Architecture section.

- [ ] Add one test per class, e.g. `test_set_source_rejects_web_authored_aws_s3_endpoint_url`: call `execute_tool("set_source", {"plugin": "aws_s3", "on_success": "t1", "options": {"bucket": "b", "key": "k", "endpoint_url": "http://127.0.0.1:4566"}, "on_validation_failure": "discard"}, state, _mock_catalog())`; assert `result.success is False` and the error names `endpoint_url`. Mirror for `set_output`/`patch_source_options`/`patch_output_options`. Add to `TestSetPipeline`: `test_set_pipeline_rejects_web_authored_aws_s3_endpoint_url_on_source` — start from `_valid_pipeline_args()`, set `args["source"]["plugin"] = "aws_s3"` and `args["source"]["options"]["endpoint_url"] = "http://127.0.0.1:4566"`, call `execute_tool("set_pipeline", args, _empty_state(), _mock_catalog())`, assert `result.success is False` and the error names `endpoint_url`; and `test_set_pipeline_rejects_web_authored_aws_s3_endpoint_url_on_output` — same but `args["outputs"][0]["plugin"] = "aws_s3"` and `args["outputs"][0]["options"]["endpoint_url"] = "http://127.0.0.1:4566"`. `_mock_catalog()`'s `get_schema` always returns a fixed `PluginSchemaInfo` regardless of the plugin name passed in, so `plugin="aws_s3"` clears the existence check with no real plugin registered — the gate must fire before `_prevalidate_source`/`_prevalidate_sink` (untouched by this task, and the reason the accept-path can't be exercised here pre-merge). No separate `apply_pipeline_recipe` test: it calls `_execute_set_pipeline` unconditionally on every code path (`sessions.py:784`), so the `set_pipeline` coverage above is exhaustive for it by construction.
- [ ] Run `pytest tests/unit/web/composer/test_tools.py -k aws_s3 -x` — fails (mutation succeeds; no gate).
- [ ] In `_execute_set_source` and `_execute_set_output`, insert:
  ```python
  endpoint_policy_error = web_aws_s3_endpoint_url_policy_error(plugin, options)
  if endpoint_policy_error is not None:
      return _failure_result(state, endpoint_policy_error)
  ```
  **`_execute_set_output` does not bind a local named `options`** — `outputs.py:134` binds `plugin`, `:141` binds `sink_options` (verified: there is no `options` local in that function). At that one anchor use `web_aws_s3_endpoint_url_policy_error(plugin, sink_options)` instead. In `_execute_patch_source_options`/`_execute_patch_output_options` use `current_source.plugin`/`new_options` and `current.plugin`/`new_options` respectively (both verified-bound locals at their anchors). In `_execute_set_pipeline`, insert the same check at all three anchors using the locals already bound there: `src_plugin`/`src_options` (sources-mapping branch, ~line 304), `src_plugin`/`legacy_src_options` (legacy source branch, ~line 355), and `out_plugin`/`out_options` (output loop, ~line 573). Import `web_aws_s3_endpoint_url_policy_error` from `elspeth.web.provider_config_policy` in `sources.py`, `outputs.py`, and `sessions.py`.
- [ ] Run the same command — PASS. Then run `pytest tests/unit/web/composer/test_tools.py -x` (full file) to confirm no regressions.
- [ ] `git add src/elspeth/web/composer/tools/sources.py src/elspeth/web/composer/tools/outputs.py src/elspeth/web/composer/tools/sessions.py tests/unit/web/composer/test_tools.py && git commit -m "feat(web): enforce aws_s3 endpoint_url gate in composer source/output tools"`

### Task 4: Guided-mode authoring parity for `aws_s3`

**Files:**
- Modify: `src/elspeth/web/composer/guided/chat_solver.py:444` (the source-resolution prompt's hardcoded valid-source list — the string literal reading `"source plugins are \`azure_blob\`, \`csv\`, \`dataverse\`, \`json\`, \`null\`, and \`text\`."`; note the phrase starts on the previous literal line, so anchor on `source plugins are`).

**Interfaces:** none — prompt text only. **Sequencing (hard):** requires plan 06 (the `aws_s3` source must exist in the catalog) AND this plan's Tasks 1–3 (the `endpoint_url` gate must already reject web-authored values), so this is the last task of this plan and must not be reordered ahead of either. Without the gate, this edit would steer guided-mode LLM authoring toward a source whose `endpoint_url` egress vector is still open — the exact Critical this plan exists to close.

- [ ] Edit the list to `\`aws_s3\`, \`azure_blob\`, \`csv\`, \`dataverse\`, \`json\`, \`null\`, and \`text\`` and append one sentence to the same prompt block: `"For \`aws_s3\`, never set \`endpoint_url\` — it is CLI/batch-only and web-authored values are rejected."` (The gate enforces this structurally; the sentence just saves the LLM a rejected round-trip.)
- [ ] No prompt-content unit test (house practice: prompt strings are verified by behavior, not pinned by string assertions). Run the guided integration suite that exercises this prompt: `pytest tests/integration/web/composer/guided/ -q`; expect no regressions.
- [ ] `git add src/elspeth/web/composer/guided/chat_solver.py && git commit -m "feat(web): let guided mode author aws_s3 sources behind the endpoint_url gate"`
