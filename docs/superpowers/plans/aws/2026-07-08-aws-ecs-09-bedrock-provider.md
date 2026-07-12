# AWS Bedrock LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Register a LiteLLM-backed `bedrock` provider in the pipeline `_PROVIDERS` registry, and add falsifiable unit tests proving (and, for cost, fixing) how the web composer's existing LiteLLM path parses Bedrock-shaped token/cost/cache metadata.

**Architecture:** `providers/bedrock.py` adds `BedrockConfig` (LiteLLM `bedrock/...` model id, optional `region_name`, no key fields — `extra="forbid"` blocks them) and `BedrockLLMProvider`, an `AuditedLLMClient` wrapper (mirrors `AzureLLMProvider`) around a tiny adapter exposing `litellm.completion()` through the OpenAI-SDK `.chat.completions.create()` shape `AuditedLLMClient` expects. It reuses the existing classification and authoritative audit path, then replaces provider detail escaping that boundary with a static error of the same ELSPETH category; no bespoke transport logic is added. On the composer side, `llm_response_parsing.py`'s `token_usage_from_response` already dedups Anthropic-family (incl. Bedrock-Claude) cache fields correctly; `_provider_cost_from_response` does not — it only reads `usage.cost` (OpenRouter's own field), never LiteLLM's provider-agnostic private response cost, so Bedrock silently reports cost as unavailable. Task 2 fixes that and pins it with real Bedrock-shaped LiteLLM responses.

Two more pieces of *shared* code need small, additive fixes now that a third provider exists: `LLMTransform.on_start()` (transform.py:1562) hardcodes a 2-way `azure_openai`/`openrouter` limiter-name ternary that silently routes Bedrock traffic into OpenRouter's rate-limit bucket, and `_classify_llm_error`'s message-substring heuristics (clients/llm.py:174-193) don't recognize litellm's real Bedrock-wrapped error text for two of five error shapes. Both are fixed in Task 1; both are additive — Azure/OpenRouter behavior is unchanged. Pipeline-side structured cache-token capture (`cache_creation_input_tokens`/`cache_read_input_tokens` landing on `TokenUsage`) stays composer-only per the design's stated scope: `AuditedLLMClient._extract_usage_from_provider_response()` only reads prompt/completion/total tokens; the full cache-token JSON still lands in the audit trail's `raw_response` blob, so nothing is silently lost — it's just absent from structured columns for pipeline (non-composer) Bedrock calls.

**Tech Stack:** Python 3.13, Pydantic v2, LiteLLM (already a composer dependency), pytest.

**Review status:** GO to implement after Plan 06 closes.

**Executable now:** **No.** Filigree `elspeth-e8dc754360` depends on Plan 06
`elspeth-7fe6aa531f`, which supplies the `aws` extra (`boto3`/`botocore`) that
installed LiteLLM 1.85.0 requires for real Bedrock calls. Plan 06 is itself
behind the shared signed-tier/Wardline baseline `elspeth-8166b310e7`, so that
operator/tooling prerequisite remains transitively enforced rather than
duplicated as a second Plan-09 edge.

**Depends on:** Plan 06. Although unit tests patch `litellm.completion`, a
registered provider is not operationally complete while its required AWS SDK
support is absent; the current environment emits missing-`botocore` warnings
and cannot make a real Bedrock call. The live test is authored here but its
mandatory task-role execution belongs to Plan 12. Touches `_PROVIDERS`,
`providers/bedrock.py`, `llm_response_parsing.py`, `composer_llm_audit.py`, and
two small additive fixes to shared code used by all three providers:
`transform.py`'s rate-limiter dispatch and `clients/llm.py`'s error classifier
(Azure/OpenRouter behavior unchanged).

**Guardrails are explicitly out of this provider slice.** Mapping a successful
LiteLLM response's `content_filter` finish reason is defensive provider error
handling; it does not configure or prove prompt-attack or harmful-content
screening. Plan 15C owns model-independent, field-level
`aws_bedrock_prompt_shield` and `aws_bedrock_content_safety` transforms using
Bedrock `ApplyGuardrail`, with independent audit, routing, IAM, and live
acceptance. Neither plan may claim the other control by implication.

**Global Constraints** (runtime-readiness spec, Bedrock LLM Readiness):
- "The `bedrock` provider is implemented through LiteLLM rather than a bespoke direct AWS SDK transport."
- "The provider must satisfy the existing `LLMProvider` protocol, preserve typed error classification, and keep raw provider responses inside the provider boundary."
- "Provider config must use AWS default credentials, optional `region_name`, and a required model identifier. It must not contain AWS access-key or secret-key fields."
- Pipeline unit tests cover "model/config validation and safe error handling ... without requiring AWS access."
- Composer: unit tests assert that Bedrock-shaped LiteLLM responses parse token, cost, and prompt-cache metadata correctly, and an explicitly selected live smoke completes one `bedrock/anthropic...` request with a region/model while credentials resolve through the ordinary AWS default chain. The test must not infer credential availability from a partial list of environment variables.

**Deviations:** The shared brief said to mirror OpenRouter's error classification. Task 1 mirrors **Azure** instead (`AuditedLLMClient`, clients/llm.py:231-465; `_classify_llm_error`, clients/llm.py:174-193 — a module-level function, not part of the `AuditedLLMClient` class, despite living in the same file) because Bedrock-via-LiteLLM is an SDK-shaped call (`litellm.completion()`), not raw HTTP like OpenRouter — Azure is the closer analog.

The original version of this section claimed empirical verification that `_classify_llm_error` correctly classifies litellm's Bedrock-shaped `RateLimitError`/`ContextWindowExceededError`/`Timeout`/`ServiceUnavailableError`/`ContentPolicyViolationError`. That claim was false. Re-verified directly against the installed litellm package's actual Bedrock exception-mapping branch (`litellm/litellm_core_utils/exception_mapping_utils.py:1002-1173`) and litellm's exception class constructors (`litellm/exceptions.py`):

- `RateLimitError` (litellm wraps as `"BedrockException: Rate Limit Error - ..."`) and `Timeout` (`"BedrockException: Timeout Error - ..."`) classify correctly **today** — their wrapper text contains `"rate limit"`/`"timeout"`, matching existing patterns. Task 1 adds tests pinning both rather than leaving them unverified.
- `ContextWindowExceededError` (`"BedrockException: Context Window Error - ..."`) and `ServiceUnavailableError` (`"BedrockException - ..."`, with `.status_code` hardcoded to `503` by litellm's own `__init__` — `exceptions.py:505-520` — regardless of the real underlying AWS status) do **not** classify correctly against the current patterns: `"context window"` is not in `CONTEXT_LENGTH_PATTERNS`, and `ServiceUnavailableError`'s message carries no digit code for `_SERVER_ERROR_CODE_PATTERN` to match. This is a genuine retry-correctness bug — a misclassified `ServiceUnavailableError` becomes a non-retryable `LLMClientError` instead of a retryable `ServerError`. Task 1 now includes a small, additive fix to both (see Task 1 Steps) instead of shipping the misclassification behind synthetic tests.
- litellm's Bedrock branch never raises `ContentPolicyViolationError` — grepped every raise site in `exception_mapping_utils.py`; the only Bedrock-adjacent one is inside the Vertex AI branch, not Bedrock's. Bedrock content-filtering surfaces via `finish_reason` on a *successful* response instead, exactly like Azure (`azure.py:207-220`). Task 1's tests cover that path, not a synthetic `ContentPolicyViolationError` mock.

---

### Task 1: `bedrock` pipeline provider registered in `_PROVIDERS`, plus the two shared-code fixes a third provider exposes

**Files:**
- Create: `src/elspeth/plugins/transforms/llm/providers/bedrock.py`
- Modify: `src/elspeth/plugins/transforms/llm/transform.py:67-68` (import), `:251-253` (`_PROVIDERS`), `:1594-1627` (`_create_provider`), `:1562` (limiter-name dispatch, 2-way → 3-way — see Steps), plus a cosmetic docstring/type-accuracy sweep at `:4-5`, `:1137-1138`, `:1322-1324`, `:1366`, `:1482` (same commit, no behavior change)
- Modify: `src/elspeth/plugins/transforms/llm/base.py:23` (class `LLMConfig`), `:66` (`provider: Literal["azure", "openrouter", "bedrock"]`), `:48` (docstring: "azure" or "openrouter" → all three)
- Modify: `src/elspeth/plugins/infrastructure/clients/llm.py:166-171` (`CONTEXT_LENGTH_PATTERNS`), `:174-193` (`_classify_llm_error` — add a `status_code`-based fallback for server errors)
- Modify: `tests/unit/plugins/llm/test_config_schema.py:95-109` (`test_explicit_union_matches` — a deliberate canary that fails when a 3rd provider registers; add `BedrockConfig`)
- Modify: `tests/unit/web/catalog/test_service.py:238,240` (`len(schema["oneOf"]) == 2` → `3`; discriminator mapping keys `{"azure", "openrouter"}` → `+ "bedrock"`) and `:284-289` (`test_llm_transform_summary_includes_provider_fields` — a **second**, previously unlisted hardcoded-provider-count breakage in the same file/class; see Steps)
- Modify: `tests/unit/plugins/llm/test_transform.py:1242` (`TestLimiterDispatch` — add the Bedrock case; see Steps)
- Modify: `tests/unit/plugins/llm/test_provider_lifecycle.py` (Bedrock config parsing, `_create_provider`, `on_start`, close/lifecycle dispatch)
- Modify: `tests/unit/plugins/llm/test_plugin_registration.py` (public config-model dispatch returns `BedrockConfig`)
- Modify: `docs/reference/configuration.md:457` (llm transform row lists bedrock) + short YAML example nearby
- Modify: `tests/golden/web/catalog/knob_schema/transform__llm.json` (regenerate after the discriminated union changes; never hand-edit)
- Test: `tests/unit/plugins/llm/test_provider_bedrock.py` (new)
- Test: `tests/unit/plugins/clients/test_llm_error_classification.py` (new `TestBedrockSpecificShapes` class — pins the classifier fix directly, independent of the Bedrock provider module)

**Interfaces:**
- Consumes: `LLMConfig` (base.py:23); `AuditedLLMClient(execution, state_id, run_id, telemetry_emit, underlying_client, *, provider, limiter, token_id, operation_id)` + `.chat_completion(model, messages, *, temperature, max_tokens, resolved_prompt_template_hash, **kwargs) -> LLMResponse` (clients/llm.py:261-410, reused unmodified — see Deviations).
- Produces: `BedrockConfig(LLMConfig)`, `BedrockLLMProvider` satisfying `LLMProvider` (provider.py:116), `_PROVIDERS["bedrock"]`.

```python
class BedrockConfig(LLMConfig):
    provider: Literal["bedrock"] = Field(default="bedrock", description="LLM provider")
    model: str = Field(..., min_length=9, max_length=512, description="LiteLLM Bedrock model id in bedrock/<id> form")
    region_name: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", description="AWS region override; default AWS region resolution otherwise")
    tracing: dict[str, Any] | None = Field(default=None, description="Tier 2 tracing (langfuse only)")

    @field_validator("model")
    @classmethod
    def _require_bedrock_prefix(cls, value: str) -> str:
        if value != value.strip() or not value.startswith("bedrock/") or not value.removeprefix("bedrock/"):
            raise ValueError("Bedrock model must be a non-empty LiteLLM 'bedrock/<model-id>' value without surrounding whitespace")
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
            raise ValueError("Bedrock model must not contain control characters")
        return value
```
No `aws_access_key_id`/`aws_secret_access_key` fields — `PluginConfig.model_config["extra"] = "forbid"` (config_base.py:163-164, inherited by `TransformDataConfig` through `DataPluginConfig`) rejects them automatically. `api_key` is likewise absent: unlike `LLMConfig`, it is not a base-class field — Azure/OpenRouter each declare it individually — so once Bedrock joins the union, `api_key` is no longer required in *every* discriminated variant (see the catalog test fix in Steps).

`BedrockLLMProvider` mirrors `AzureLLMProvider` (azure.py:101-300) for client caching, `execute_query`'s finish_reason/empty-content handling, and `close()`. It differs only in `_get_underlying_client()`, which builds an adapter instead of `AzureOpenAI(...)`:
```python
class _LiteLLMSDKAdapter:
    """Exposes litellm.completion() as client.chat.completions.create(**kw)."""
    def __init__(self, *, region_name: str | None) -> None:
        self._region_name = region_name
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> Any:
        import litellm
        if self._region_name is not None:
            kwargs.setdefault("aws_region_name", self._region_name)
        return litellm.completion(**kwargs)

    def close(self) -> None:
        pass  # litellm.completion() is stateless per call
```
and passes `provider="bedrock"` (not `"azure"`) into both `_get_llm_client` and `runtime_preflight`'s `AuditedLLMClient(...)`; drops `endpoint`/`api_key`/`api_version`/`deployment_name` ctor args (Bedrock has none).

The adapter is immutable after construction and adds only `aws_region_name` to
the per-call kwargs copy. It must never add access-key, secret-key,
session-token, profile, role, endpoint, or API-key kwargs; boto3/LiteLLM resolve
credentials from the ordinary AWS default chain. Unit tests pin the exact
forwarding contract. Do not add an unenforced "keep pool size conservative"
warning: the provider supports the existing pool contract, and any discovered
concurrency defect must become a tested implementation issue rather than
operator folklore.

**Steps:**
- [ ] Write `TestBedrockSpecificShapes` in `tests/unit/plugins/clients/test_llm_error_classification.py` (mirrors `TestAzureSpecificCodes`, same file): `test_bedrock_context_window_wrapper_is_context_length` (`_classify_llm_error(Exception("BedrockException: Context Window Error - Input is too long for requested model."))` must return `"context_length"` — litellm's real wrapper for 5 of 6 Bedrock context-overflow triggers, `exception_mapping_utils.py:1002-1016`); `test_bedrock_service_unavailable_hardcoded_503_is_server` (a local `_StatusCodeError(Exception)` with `__init__(self, message, status_code)` setting `self.status_code = status_code`, constructed as `_StatusCodeError("BedrockException - Internal server error", status_code=503)`, must classify `"server"` — litellm's `ServiceUnavailableError.__init__` hardcodes `self.status_code = 503` unconditionally, `exceptions.py:505-520`, and the message carries no digit).
- [ ] Run `uv run pytest tests/unit/plugins/clients/test_llm_error_classification.py -v` → both new tests fail (`'unknown' != 'context_length'` / `'unknown' != 'server'`).
- [ ] Add `"context window"` to `CONTEXT_LENGTH_PATTERNS` (clients/llm.py:166-171). For the server fallback, read `status_code` from the exception's instance-data mapping (`vars(exception)`) rather than invoking a provider property with `getattr`; require `type(status_code) is int` before membership in `(500, 502, 503, 504, 529)`. Add cases for `None`, `True`, string `"503"`, list, dict, and a raising `status_code` property; none may raise or classify as server. The check runs after context/rate-limit precedence and therefore does not intercept LiteLLM's context error (`status_code=400`).
- [ ] Run the same command → all pass, including the pre-existing suite (no regression).
- [ ] Write `test_provider_bedrock.py`: validate the prefix, non-empty suffix, surrounding whitespace/control-character rejection, 512-character bound, valid omitted/explicit region, invalid/empty/oversized region, and parameterized rejection of `api_key`, AWS access-key, secret-key, session-token, profile, role, and endpoint fields. Assert the provider satisfies `LLMProvider`. Adapter-path tests patch `litellm.completion` and prove model/messages/temperature/max-tokens survive unchanged, configured `region_name` becomes `aws_region_name`, omitted region adds no AWS kwarg, and no credential/endpoint kwarg is forwarded. Error-classification and success-path tests use LiteLLM's real exception/`ModelResponse` construction, not idealized text:
  - `test_execute_query_classifies_rate_limit_error`: `litellm.exceptions.RateLimitError(message="BedrockException: Rate Limit Error - ThrottlingException: Rate exceeded", llm_provider="bedrock", model=<bedrock model id>)` → `RateLimitError`, `retryable=True`.
  - `test_execute_query_classifies_context_length_error`: `litellm.exceptions.ContextWindowExceededError(message="BedrockException: Context Window Error - Input is too long for requested model.", model=<id>, llm_provider="bedrock")` → `ContextLengthError`, `retryable=False` (depends on the classifier fix above).
  - `test_execute_query_classifies_service_unavailable_error` (new): `litellm.exceptions.ServiceUnavailableError(message="BedrockException - Internal server error", llm_provider="bedrock", model=<id>, response=httpx.Response(status_code=500, request=httpx.Request("POST", "https://api.openai.com/v1/")))` → `ServerError`, `retryable=True` (depends on the classifier fix above; litellm hardcodes `.status_code=503` on this class regardless of the passed `response`).
  - `test_execute_query_classifies_timeout_error` (new): `litellm.exceptions.Timeout(message="BedrockException: Timeout Error - Connect timeout on endpoint URL", model=<id>, llm_provider="bedrock")` → `NetworkError`, `retryable=True` (already correct today — this pins it against regression).
  - `test_execute_query_content_filter_raises_content_policy_error` (replaces a synthetic `ContentPolicyViolationError` mock — litellm's Bedrock branch never raises that class): mirrors `test_provider_azure.py`'s `test_empty_content_raises_content_policy_error` (:378) — `patch("litellm.completion", return_value=litellm.types.utils.ModelResponse(choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "content_filter"}], model=<id>, usage={...}))` → `ContentPolicyError`, `match="empty content"`.
  - `test_execute_query_returns_llm_query_result_on_success` (new — closes the plan's previously-untested success path through `_LiteLLMSDKAdapter`): `patch("litellm.completion", return_value=<ModelResponse with populated content/usage/model>)` → asserts `LLMQueryResult.content`/`.usage`/`.model` match, mirroring `test_provider_azure.py`'s `test_returns_llm_query_result` (:189).
- [ ] Add a pipeline redaction regression with a unique AWS ARN/account/request-id/credential-shaped sentinel in the raw LiteLLM error. `AuditedLLMClient` may retain the raw error only in its authoritative audit record; `BedrockLLMProvider.execute_query` and `runtime_preflight` must re-raise the same ELSPETH typed category with a static Bedrock message outside the active `except` block, so the escaping exception/result has no raw cause/context and none of the sentinels. Prove retryability is preserved for rate/server/network categories. This is the pipeline half of the spec's safe-error contract; the composer test alone is insufficient.
- [ ] Run `uv run pytest tests/unit/plugins/llm/test_provider_bedrock.py -v` → fails: `ModuleNotFoundError: No module named 'elspeth.plugins.transforms.llm.providers.bedrock'`.
- [ ] Implement `bedrock.py`; wire `_PROVIDERS["bedrock"] = (BedrockConfig, BedrockLLMProvider)`, the `_create_provider` `elif isinstance(self._config, BedrockConfig): return BedrockLLMProvider(region_name=self._config.region_name, recorder=self._recorder, run_id=self._run_id, telemetry_emit=self._telemetry_emit, limiter=self._limiter, resolved_prompt_template_hash=self._resolved_prompt_template_hash)`, and the base Literal.
- [ ] Run `uv run pytest tests/unit/plugins/llm/test_provider_bedrock.py -v` → all pass.
- [ ] Extend `test_provider_lifecycle.py` and `test_plugin_registration.py` for the third provider. Prove the parsed config type, public config-model lookup, provider construction after recorder initialization, `on_start`, protocol shape, and close path; this prevents a registry-only test from missing lifecycle dispatch.
- [ ] First extend `tests/unit/plugins/llm/test_transform.py::_make_config` with an explicit Bedrock branch and a valid `bedrock/<model-id>`; without it the proposed limiter test dies during config construction. Then write `test_bedrock_provider_gets_bedrock_limiter` in `TestLimiterDispatch`, call `on_start()` with `_RegistryDouble()`, and assert `get_limiter("bedrock")`.
- [ ] Run `uv run pytest tests/unit/plugins/llm/test_transform.py -k TestLimiterDispatch -v` → the new test fails: `get_limiter` was called with `"openrouter"`, not `"bedrock"` (transform.py:1562's `azure_openai`/`openrouter` ternary has no third branch, so a `BedrockConfig` — not an `isinstance(..., AzureOpenAIConfig)` — falls into the `else`).
- [ ] Change transform.py:1562 from a binary ternary to a 3-way dispatch: `limiter_name = "azure_openai" if isinstance(self._config, AzureOpenAIConfig) else "bedrock" if isinstance(self._config, BedrockConfig) else "openrouter"`. While the file is open, update the stale provider-set docstrings/comments/types the review found: module docstring (`:4-5`), the `LLMTransform` class docstring's "Provider dispatch" table (`:1137-1138`, add `"bedrock" → BedrockConfig + BedrockLLMProvider"`), the tracing comment (`:1366`, "AzureOpenAIConfig, OpenRouterConfig" → add `BedrockConfig`), `provider_config`'s return type and its `cast(...)` call (`:1322-1324`, `:1482`, → `AzureOpenAIConfig | OpenRouterConfig | BedrockConfig`), and `base.py:48`'s "azure" or "openrouter" docstring line. Cosmetic/type-accuracy only, no behavior change.
- [ ] Run `uv run pytest tests/unit/plugins/llm/test_transform.py -k TestLimiterDispatch -v` → all three tests (azure/openrouter/bedrock) pass.
- [ ] Run `uv run pytest tests/unit/plugins/llm/test_config_schema.py -v` → `test_explicit_union_matches` fails (fixture still 2-provider). Add `BedrockConfig` to its union and to the `("AzureOpenAIConfig", "OpenRouterConfig")` name tuple.
- [ ] Run `uv run pytest tests/unit/plugins/llm/ tests/unit/web/catalog/test_service.py -v` → the hard-coded two-provider schema assertions fail. Update oneOf/discriminator expectations to three providers, keep `prompt_template` required, and change `api_key` to not-universally-required because Bedrock intentionally has no key field. Extend catalog field-ordering coverage for the Bedrock-only `region_name` field rather than merely changing counts.
- [ ] Run `uv run pytest tests/unit/plugins/llm/ tests/unit/web/catalog/test_service.py -v` → all pass.
- [ ] After the final `transform.py` edit, recompute `LLMTransform.source_file_hash` exactly once with the repository helper:

  ```bash
  uv run python -c "from pathlib import Path; from scripts.cicd.plugin_hash import compute_source_file_hash, fix_source_file_hash; p=Path('src/elspeth/plugins/transforms/llm/transform.py'); fix_source_file_hash(p, 'LLMTransform', compute_source_file_hash(p))"
  ```

  Regenerate only the LLM golden with this deterministic command (the golden pytest is compare-only; there is no repository update flag):

  ```bash
  uv run python - <<'PY'
  import json
  from pathlib import Path
  from elspeth.plugins.infrastructure.manager import get_shared_plugin_manager
  from elspeth.web.catalog.service import CatalogServiceImpl

  service = CatalogServiceImpl(get_shared_plugin_manager())
  info = service._schema_cache[("transform", "llm")]
  payload = {"plugin_kind": "transform", "plugin_name": "llm", "knob_schema": info.knob_schema}
  Path("tests/golden/web/catalog/knob_schema/transform__llm.json").write_text(
      json.dumps(payload, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  PY
  ```

  Run the focused golden and `plugin_contract.component_type,plugin_contract.plugin_hashes` gates; never hand-edit the hash or golden JSON.
- [ ] Add bedrock to the `llm` row (configuration.md:457) and a short YAML example (`provider: bedrock`, `model: bedrock/anthropic...`, `region_name`) noting ECS task-role Bedrock permissions replace access keys.
- [ ] Stage only the exact Task-1 files (including the regenerated golden); never `git add` the whole LLM directory. Commit `feat(llm): add LiteLLM-backed bedrock provider to _PROVIDERS` only after the focused plugin/hash/golden tests pass.

### Task 2: fix composer provider-cost extraction for non-OpenRouter (Bedrock) responses

**Files:**
- Modify: `src/elspeth/contracts/composer_llm_audit.py:23-27`
- Modify: `src/elspeth/web/composer/llm_response_parsing.py:181-201` (`_provider_cost_from_response`)
- Test: `tests/unit/web/composer/test_llm_response_parsing_bedrock.py` (new)
- Modify: `tests/unit/contracts/test_composer_llm_audit.py` (new cost-source contract case)
- Modify: `tests/unit/web/composer/test_compose_loop_llm_audit.py` (retain the no-dynamic-`getattr` parser invariant)

**Interfaces:**
- Consumes: `token_usage_from_response`, `_provider_cost_from_response`, and `build_llm_call_record`. Installed LiteLLM 1.85.0 `ModelResponse` stores `_hidden_params` in its Pydantic `__pydantic_private__` mapping, not `vars(response)` or `model_dump()`.
- Produces: `PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST: ComposerLLMProviderCostSource = "_hidden_params.response_cost"`.

```python
ComposerLLMProviderCostSource = Literal["not_available", "response_usage.cost", "_hidden_params.response_cost"]
PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST: ComposerLLMProviderCostSource = "_hidden_params.response_cost"
_VALID_PROVIDER_COST_SOURCES = {PROVIDER_COST_SOURCE_NOT_AVAILABLE, PROVIDER_COST_SOURCE_RESPONSE_USAGE_COST, PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST}
```
In `_provider_cost_from_response`, fall back to LiteLLM's private
`response_cost` only when `usage.cost` is **absent**. If the higher-priority
`usage.cost` is present but malformed, boolean, negative, or non-finite, return
unavailable rather than masking corrupt primary metadata. Add a dedicated
data-only helper that reads `__pydantic_private__` with
`object.__getattribute__`, validates each layer as a `Mapping`, and never
invokes a provider-named property. Preserve the module's enforced no-dynamic-
`getattr` rule. LiteLLM's installed `ResponseMetadata.set_hidden_params`
computes the private value; this is a pinned private API and therefore needs a
real `ModelResponse` regression.

**Steps:**
- [ ] Write `test_llm_response_parsing_bedrock.py` around an actual `litellm.types.utils.ModelResponse`; assign `response._hidden_params = {"response_cost": 0.01234}` after construction. Assert token/cache sibling dedup, hidden-cost extraction, OpenRouter `usage.cost` precedence, zero acceptance, and hidden boolean/string/negative/NaN/inf rejection. Assert a present invalid `usage.cost` remains unavailable even when the hidden value is valid. Add a malicious response whose `_hidden_params` property raises and prove the parser never invokes it.
- [ ] Run the focused file: the valid private-cost and record-persistence cases fail before implementation; existing-unavailable pins may already pass and must not be misreported as red.
- [ ] Add the literal/constant and data-only Pydantic-private helper, with hidden fallback only after confirmed absence of `usage.cost`.
- [ ] Test through `build_llm_call_record`, asserting persisted `provider_cost`, `_hidden_params.response_cost` provenance, cache counters, returned model, and provider request id. In `test_composer_llm_audit.py`, construct/serialize a `ComposerLLMCall` with the new source so `_VALID_PROVIDER_COST_SOURCES` is covered at the contract layer.
- [ ] Run the Bedrock parser file, composer audit contract tests, the parser no-dynamic-`getattr` invariant, and the existing provider-cache audit class.
- [ ] Stage the five exact Task-2 source/test files and commit `fix(composer): read LiteLLM private response cost for Bedrock`.

### Task 3: Bedrock provider-error redaction pin + opt-in live smoke test

**Files:**
- Modify: `tests/unit/web/composer/test_service.py` (new test near line 2989)
- Create: `tests/integration/web/composer/test_bedrock_live_smoke.py`

**Interfaces:**
- Consumes: `ComposerServiceImpl.compose`, `_litellm_acompletion` (currently `service.py:389`), `litellm.exceptions.BadRequestError`; `_make_settings`/`_mock_catalog`/`_empty_state` (`_helpers.py`); `integration`/`slow` markers and pytest-timeout.

**Steps:**
- [ ] Write `test_bedrock_bad_request_raises_redacted_service_error` in `test_service.py`, mirroring the existing LiteLLM case. Put unique raw-detail, AWS ARN/account/request-id, URL, and credential-shaped sentinels in the provider exception; assert the public message remains exactly `LLM request rejected (BadRequestError)` and no sentinel appears in `str`, `repr`, logs, or response-visible audit error fields. The authoritative internal audit may retain only the already-governed hashed/redacted representation.
- [ ] Run `uv run pytest tests/unit/web/composer/test_service.py -k bedrock -v` → passes immediately: redaction keys on exception type, not provider name, so this pins correct behavior against regression rather than fixing a bug.
- [ ] Write `test_bedrock_live_smoke.py` with `integration`, `slow`, `asyncio`, and `timeout(60)` markers. Gate only on explicit operator selection `ELSPETH_RUN_BEDROCK_LIVE=1`; when selected, require a `bedrock/` `ELSPETH_BEDROCK_LIVE_TEST_MODEL` and region from `AWS_REGION` or `AWS_DEFAULT_REGION`, then await the composer's `_litellm_acompletion` with a one-line message and `max_tokens=16`. Do **not** inspect access-key/profile/container environment variables: ECS task role, web identity, IMDS, SSO, and other legitimate boto default-chain modes are not enumerable here. An explicitly selected test with unavailable credentials/permissions fails honestly. Assert non-empty content and run the normal token/cost/cache/model/request-id parser over the response.
- [ ] Run the file with no opt-in → one declared skip. Plan 12 repeats the equivalent production acceptance helper **inside the candidate ECS task** with the task role and requires success; a controller-side skip is never release evidence.
- [ ] `git add tests/unit/web/composer/test_service.py tests/integration/web/composer/test_bedrock_live_smoke.py && git commit -m "test(composer): pin Bedrock error redaction and add opt-in live Bedrock smoke test"`

**Acceptance handoff:** Task 3 authors a developer/operator smoke, but the ECS
default-chain claim is accepted only by Plans 10/12 executing the production
acceptance helper inside the candidate task. Missing model, region, permission,
content, or task-role credentials is BLOCKED/NO-GO there, not a skip.

---

### Task 4: integrated static, signed-trust, and Wardline handoff

**Files:** no planned product changes. Operator-owned signed allowlist YAML may
change only after read-only diagnosis and operator review/signing; never stage
backup files.

- [ ] Require Plan 06 and the transitive gate-baseline issue closed before
  starting. Verify the frozen dependency surface:

  ```bash
  uv lock --check
  uv sync --frozen --all-extras
  uv run python -c "import boto3, botocore, litellm"
  ```

- [ ] Verify the Task-1 hash and generated `transform__llm.json`; do not
  regenerate either merely because Task 4 started. If a gate correction touches
  `transform.py` or the LLM schema, return to Task 1, recompute/regenerate once,
  and restart all Task-4 gates. Then run:

  ```bash
  uv run pytest tests/unit/plugins/llm/ tests/unit/plugins/clients/test_llm_error_classification.py tests/unit/contracts/test_composer_llm_audit.py tests/unit/web/catalog/test_service.py tests/unit/web/catalog/test_knob_schema_golden.py tests/unit/web/composer/test_llm_response_parsing_bedrock.py tests/unit/web/composer/test_service.py tests/unit/web/composer/test_compose_loop_llm_audit.py -q
  uv run pytest tests/integration/web/composer/test_bedrock_live_smoke.py -m "slow and integration" --collect-only -q
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.options_metadata --root .
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules plugin_contract.component_type,plugin_contract.plugin_hashes --root src/elspeth
  ```

- [ ] Run the repository static gates:

  ```bash
  uv run ruff check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run ruff format --check src/ tests/ scripts/ examples/ elspeth-lints/src/
  uv run mypy src/ elspeth-lints/src/
  ```

- [ ] Run read-only signed-entry diagnosis. Require every entry to be
  `OK`/`OK_SHAPE_ONLY`:

  ```bash
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints diagnose-judge-signatures --root src/elspeth --allowlist-dir config/cicd/enforce_tier_model --format text
  ELSPETH_JUDGE_METADATA_SIGNATURE_VERIFY_MODE=shape-only-when-key-missing uv run elspeth-lints check --rules trust_tier.tier_model --root src/elspeth
  PYTHONPATH=elspeth-lints/src uv run python -m elspeth_lints.core.cli check --rules trust_boundary.tests,trust_boundary.scope,trust_boundary.tier --root src/elspeth
  ```

  Any binding drift stops the agent. Only the operator may review a dry run and
  repair the diagnosed rows with the operator-held key. The agent never runs a
  signing command, receives the key, hand-edits an allowlist, or stages
  `*.bak-*`; after the operator change, rerun diagnosis and both trust gates.

- [ ] If and only if the operator repair changed signed YAML, inspect the diff
  and stage **only** the exact diagnosed files named in the operator handoff:

  ```bash
  git diff -- config/cicd/enforce_tier_model
  test -z "$(find config/cicd/enforce_tier_model -type f -name '*.bak-*' -print -quit)"
  git add -- config/cicd/enforce_tier_model/<diagnosed-file-1>.yaml [config/cicd/enforce_tier_model/<diagnosed-file-2>.yaml ...]
  git diff --cached --name-only
  git commit -m "chore(cicd): refresh signed bindings for Bedrock provider"
  ```

  The staged list must equal the operator's diagnosed YAML list and contain no
  source, plan, unrelated allowlist, or backup file. Do not create an empty
  commit. After this commit, rerun signed diagnosis, both trust gates, the
  focused tests, plugin hash/golden checks, and static gates on the new HEAD.

- [ ] Run `git diff --check` and `wardline scan . --fail-on ERROR`. Wardline
  exit 1 requires explain/fix/rescan at the external-input boundary; exit 2 is
  also blocking. No baseline, waiver, narrowed substitute, or degraded scan.
- [ ] Close Plan 09 only after every local gate above passes. The opt-in local
  smoke may remain skipped in ordinary execution, but Plan 10 must own an
  in-image `verify-bedrock` acceptance command and Plan 12 must execute it
  inside the candidate ECS task with zero skips/failures before runtime GO.

**Definition of Done:** the third provider is registered and lifecycle-tested;
Bedrock config/credential boundaries are fail closed; pipeline and composer
errors are redacted; real LiteLLM private cost/cache/token metadata persists
with honest provenance; plugin hash/golden/static/trust/Wardline gates pass;
and the Plan-10/12 task-role acceptance handoff is explicit.
