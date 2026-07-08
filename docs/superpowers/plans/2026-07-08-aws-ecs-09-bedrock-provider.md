# AWS Bedrock LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Register a LiteLLM-backed `bedrock` provider in the pipeline `_PROVIDERS` registry, and add falsifiable unit tests proving (and, for cost, fixing) how the web composer's existing LiteLLM path parses Bedrock-shaped token/cost/cache metadata.

**Architecture:** `providers/bedrock.py` adds `BedrockConfig` (LiteLLM `bedrock/...` model id, optional `region_name`, no key fields — `extra="forbid"` blocks them) and `BedrockLLMProvider`, an `AuditedLLMClient` wrapper (mirrors `AzureLLMProvider`) around a tiny adapter exposing `litellm.completion()` through the OpenAI-SDK `.chat.completions.create()` shape `AuditedLLMClient` expects — reusing its error classification and audit recording verbatim, no bespoke transport logic. On the composer side, `llm_response_parsing.py`'s `token_usage_from_response` already dedups Anthropic-family (incl. Bedrock-Claude) cache fields correctly; `_provider_cost_from_response` does not — it only reads `usage.cost` (OpenRouter's own field), never LiteLLM's provider-agnostic `response._hidden_params["response_cost"]`, so Bedrock silently reports cost as unavailable. Task 2 fixes that and pins it with Bedrock-shaped fixtures.

Two more pieces of *shared* code need small, additive fixes now that a third provider exists: `LLMTransform.on_start()` (transform.py:1562) hardcodes a 2-way `azure_openai`/`openrouter` limiter-name ternary that silently routes Bedrock traffic into OpenRouter's rate-limit bucket, and `_classify_llm_error`'s message-substring heuristics (clients/llm.py:174-193) don't recognize litellm's real Bedrock-wrapped error text for two of five error shapes. Both are fixed in Task 1; both are additive — Azure/OpenRouter behavior is unchanged. Pipeline-side structured cache-token capture (`cache_creation_input_tokens`/`cache_read_input_tokens` landing on `TokenUsage`) stays composer-only per the design's stated scope: `AuditedLLMClient._extract_usage_from_provider_response()` only reads prompt/completion/total tokens; the full cache-token JSON still lands in the audit trail's `raw_response` blob, so nothing is silently lost — it's just absent from structured columns for pipeline (non-composer) Bedrock calls.

**Tech Stack:** Python 3.13, Pydantic v2, LiteLLM (already a composer dependency), pytest.

**Depends on:** None. Touches `_PROVIDERS`, `providers/bedrock.py`, `llm_response_parsing.py`, `composer_llm_audit.py`, and two small additive fixes to already-shared code used by all three providers: `transform.py`'s rate-limiter dispatch and `clients/llm.py`'s error classifier (Azure/OpenRouter behavior unchanged). Unit tests patch `litellm.completion`, so the `aws` extra (boto3/botocore) is needed only for the opt-in live smoke test.

**Global Constraints** (verbatim spec, Bedrock LLM Readiness):
- "The `bedrock` provider is implemented through LiteLLM rather than a bespoke direct AWS SDK transport."
- "The provider must satisfy the existing `LLMProvider` protocol, preserve typed error classification, and keep raw provider responses inside the provider boundary."
- "Provider config must use AWS default credentials, optional `region_name`, and a required model identifier. It must not contain AWS access-key or secret-key fields."
- Pipeline unit tests cover "model/config validation and safe error handling ... without requiring AWS access."
- Composer: "unit tests assert that Bedrock-shaped LiteLLM responses parse token, cost, and prompt-cache metadata correctly, and an opt-in live smoke test completes one `bedrock/anthropic...` request when explicit AWS credentials, region, and model settings are present."

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
- Modify: `docs/reference/configuration.md:457` (llm transform row lists bedrock) + short YAML example nearby
- Test: `tests/unit/plugins/llm/test_provider_bedrock.py` (new)
- Test: `tests/unit/plugins/clients/test_llm_error_classification.py` (new `TestBedrockSpecificShapes` class — pins the classifier fix directly, independent of the Bedrock provider module)

**Interfaces:**
- Consumes: `LLMConfig` (base.py:23); `AuditedLLMClient(execution, state_id, run_id, telemetry_emit, underlying_client, *, provider, limiter, token_id, operation_id)` + `.chat_completion(model, messages, *, temperature, max_tokens, resolved_prompt_template_hash, **kwargs) -> LLMResponse` (clients/llm.py:261-410, reused unmodified — see Deviations).
- Produces: `BedrockConfig(LLMConfig)`, `BedrockLLMProvider` satisfying `LLMProvider` (provider.py:116), `_PROVIDERS["bedrock"]`.

```python
class BedrockConfig(LLMConfig):
    provider: Literal["bedrock"] = Field(default="bedrock", description="LLM provider")
    model: str = Field(..., description="LiteLLM Bedrock model id, e.g. 'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0'")
    region_name: str | None = Field(default=None, description="AWS region override; default AWS region resolution otherwise")
    tracing: dict[str, Any] | None = Field(default=None, description="Tier 2 tracing (langfuse only)")

    @field_validator("model")
    @classmethod
    def _require_bedrock_prefix(cls, value: str) -> str:
        if not value.startswith("bedrock/"):
            raise ValueError(f"Bedrock model must use the LiteLLM 'bedrock/...' form, got {value!r}")
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

Unlike Azure/OpenRouter, which each hold a per-provider, lazily-built, lock-guarded client instance (`_get_underlying_client()`/`_underlying_client_lock`, azure.py:261-274), `_LiteLLMSDKAdapter.create()` calls the shared module-level `litellm.completion()` function directly on every invocation. Concurrent pool workers (`pool_size > 1`) all call through the same module function rather than a per-provider guarded instance; no concrete race has been found or reproduced in litellm itself, so this is a plausibility note, not a confirmed defect. Keep Bedrock `pool_size` conservative until litellm's concurrent-invocation safety is independently confirmed.

**Steps:**
- [ ] Write `TestBedrockSpecificShapes` in `tests/unit/plugins/clients/test_llm_error_classification.py` (mirrors `TestAzureSpecificCodes`, same file): `test_bedrock_context_window_wrapper_is_context_length` (`_classify_llm_error(Exception("BedrockException: Context Window Error - Input is too long for requested model."))` must return `"context_length"` — litellm's real wrapper for 5 of 6 Bedrock context-overflow triggers, `exception_mapping_utils.py:1002-1016`); `test_bedrock_service_unavailable_hardcoded_503_is_server` (a local `_StatusCodeError(Exception)` with `__init__(self, message, status_code)` setting `self.status_code = status_code`, constructed as `_StatusCodeError("BedrockException - Internal server error", status_code=503)`, must classify `"server"` — litellm's `ServiceUnavailableError.__init__` hardcodes `self.status_code = 503` unconditionally, `exceptions.py:505-520`, and the message carries no digit).
- [ ] Run `pytest tests/unit/plugins/clients/test_llm_error_classification.py -v` → both new tests fail (`'unknown' != 'context_length'` / `'unknown' != 'server'`).
- [ ] Add `"context window"` to `CONTEXT_LENGTH_PATTERNS` (clients/llm.py:166-171), and in `_classify_llm_error`'s server-error check (clients/llm.py:187-188) also match `getattr(exception, "status_code", None) in {500, 502, 503, 504, 529}` before falling through. Purely additive — existing message-text paths for Azure/OpenRouter are untouched, and this check runs after the context-length/rate-limit checks so it can never intercept `ContextWindowExceededError` (which litellm hardcodes to `status_code=400`, outside the server set).
- [ ] Run the same command → all pass, including the pre-existing suite (no regression).
- [ ] Write `test_provider_bedrock.py`: `test_config_requires_bedrock_prefixed_model` (`model="anthropic.claude-3"` → `ValidationError` matching `"bedrock/...' form"`), `test_config_rejects_access_key_fields` (`aws_access_key_id="AKIA..."` → `ValidationError` matching `"extra"`), `test_config_region_name_optional`, `test_provider_satisfies_llmprovider_protocol` (`isinstance(BedrockLLMProvider(...), LLMProvider)`), with `FakeAuditRecorder`/`FakeTelemetryEmit` mirroring `test_provider_azure.py`. Error-classification and success-path tests use litellm's real construction (`patch("litellm.completion", side_effect=<exc>)` / `return_value=<litellm.types.utils.ModelResponse(...)>`), not idealized text:
  - `test_execute_query_classifies_rate_limit_error`: `litellm.exceptions.RateLimitError(message="BedrockException: Rate Limit Error - ThrottlingException: Rate exceeded", llm_provider="bedrock", model=<bedrock model id>)` → `RateLimitError`, `retryable=True`.
  - `test_execute_query_classifies_context_length_error`: `litellm.exceptions.ContextWindowExceededError(message="BedrockException: Context Window Error - Input is too long for requested model.", model=<id>, llm_provider="bedrock")` → `ContextLengthError`, `retryable=False` (depends on the classifier fix above).
  - `test_execute_query_classifies_service_unavailable_error` (new): `litellm.exceptions.ServiceUnavailableError(message="BedrockException - Internal server error", llm_provider="bedrock", model=<id>, response=httpx.Response(status_code=500, request=httpx.Request("POST", "https://api.openai.com/v1/")))` → `ServerError`, `retryable=True` (depends on the classifier fix above; litellm hardcodes `.status_code=503` on this class regardless of the passed `response`).
  - `test_execute_query_classifies_timeout_error` (new): `litellm.exceptions.Timeout(message="BedrockException: Timeout Error - Connect timeout on endpoint URL", model=<id>, llm_provider="bedrock")` → `NetworkError`, `retryable=True` (already correct today — this pins it against regression).
  - `test_execute_query_content_filter_raises_content_policy_error` (replaces a synthetic `ContentPolicyViolationError` mock — litellm's Bedrock branch never raises that class): mirrors `test_provider_azure.py`'s `test_empty_content_raises_content_policy_error` (:378) — `patch("litellm.completion", return_value=litellm.types.utils.ModelResponse(choices=[{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "content_filter"}], model=<id>, usage={...}))` → `ContentPolicyError`, `match="empty content"`.
  - `test_execute_query_returns_llm_query_result_on_success` (new — closes the plan's previously-untested success path through `_LiteLLMSDKAdapter`): `patch("litellm.completion", return_value=<ModelResponse with populated content/usage/model>)` → asserts `LLMQueryResult.content`/`.usage`/`.model` match, mirroring `test_provider_azure.py`'s `test_returns_llm_query_result` (:189).
- [ ] Run `pytest tests/unit/plugins/llm/test_provider_bedrock.py -v` → fails: `ModuleNotFoundError: No module named 'elspeth.plugins.transforms.llm.providers.bedrock'`.
- [ ] Implement `bedrock.py`; wire `_PROVIDERS["bedrock"] = (BedrockConfig, BedrockLLMProvider)`, the `_create_provider` `elif isinstance(self._config, BedrockConfig): return BedrockLLMProvider(region_name=self._config.region_name, recorder=self._recorder, run_id=self._run_id, telemetry_emit=self._telemetry_emit, limiter=self._limiter, resolved_prompt_template_hash=self._resolved_prompt_template_hash)`, and the base Literal.
- [ ] Run `pytest tests/unit/plugins/llm/test_provider_bedrock.py -v` → all pass.
- [ ] Write `test_bedrock_provider_gets_bedrock_limiter` in `TestLimiterDispatch` (`test_transform.py:1242`, mirroring `test_openrouter_provider_gets_openrouter_limiter` at :1267): construct `LLMTransform(_make_config(provider="bedrock"))`, call `on_start()` with a `_RegistryDouble()`, assert `mock_registry.get_limiter.assert_called_once_with("bedrock")`.
- [ ] Run `pytest tests/unit/plugins/llm/test_transform.py -k TestLimiterDispatch -v` → the new test fails: `get_limiter` was called with `"openrouter"`, not `"bedrock"` (transform.py:1562's `azure_openai`/`openrouter` ternary has no third branch, so a `BedrockConfig` — not an `isinstance(..., AzureOpenAIConfig)` — falls into the `else`).
- [ ] Change transform.py:1562 from a binary ternary to a 3-way dispatch: `limiter_name = "azure_openai" if isinstance(self._config, AzureOpenAIConfig) else "bedrock" if isinstance(self._config, BedrockConfig) else "openrouter"`. While the file is open, update the stale provider-set docstrings/comments/types the review found: module docstring (`:4-5`), the `LLMTransform` class docstring's "Provider dispatch" table (`:1137-1138`, add `"bedrock" → BedrockConfig + BedrockLLMProvider"`), the tracing comment (`:1366`, "AzureOpenAIConfig, OpenRouterConfig" → add `BedrockConfig`), `provider_config`'s return type and its `cast(...)` call (`:1322-1324`, `:1482`, → `AzureOpenAIConfig | OpenRouterConfig | BedrockConfig`), and `base.py:48`'s "azure" or "openrouter" docstring line. Cosmetic/type-accuracy only, no behavior change.
- [ ] Run `pytest tests/unit/plugins/llm/test_transform.py -k TestLimiterDispatch -v` → all three tests (azure/openrouter/bedrock) pass.
- [ ] Run `pytest tests/unit/plugins/llm/test_config_schema.py -v` → `test_explicit_union_matches` fails (fixture still 2-provider). Add `BedrockConfig` to its union and to the `("AzureOpenAIConfig", "OpenRouterConfig")` name tuple.
- [ ] Run `pytest tests/unit/plugins/llm/ tests/unit/web/catalog/test_service.py -v` → two catalog tests fail. `test_llm_schema_...` (`test_service.py:238,240`): `len(oneOf) == 2` / mapping keys missing `"bedrock"` — update to `3` and `{"azure", "openrouter", "bedrock"}`. `test_llm_transform_summary_includes_provider_fields` (`:284-289`): `assert "api_key" in required` — `api_key` lives on `AzureOpenAIConfig`/`OpenRouterConfig` individually (not on base `LLMConfig`), `BedrockConfig` has no `api_key` field, so `_fields_from_discriminated`'s "required iff required in every variant" rule (`web/catalog/service.py:~563`) correctly drops it from the honest intersection once Bedrock registers — change to `assert "api_key" not in required` with a one-line comment explaining Bedrock breaks the previous every-variant invariant. `prompt_template` stays `in required` unchanged — it's declared on the shared `LLMConfig` base, so all three variants still require it.
- [ ] Run `pytest tests/unit/plugins/llm/ tests/unit/web/catalog/test_service.py -v` → all pass.
- [ ] Add bedrock to the `llm` row (configuration.md:457) and a short YAML example (`provider: bedrock`, `model: bedrock/anthropic...`, `region_name`) noting ECS task-role Bedrock permissions replace access keys.
- [ ] `git add src/elspeth/plugins/transforms/llm/ src/elspeth/plugins/infrastructure/clients/llm.py tests/unit/plugins/llm/test_provider_bedrock.py tests/unit/plugins/llm/test_config_schema.py tests/unit/plugins/llm/test_transform.py tests/unit/plugins/clients/test_llm_error_classification.py tests/unit/web/catalog/test_service.py docs/reference/configuration.md && git commit -m "feat(llm): add LiteLLM-backed bedrock provider to _PROVIDERS"`

### Task 2: fix composer provider-cost extraction for non-OpenRouter (Bedrock) responses

**Files:**
- Modify: `src/elspeth/contracts/composer_llm_audit.py:23-27`
- Modify: `src/elspeth/web/composer/llm_response_parsing.py:181-201` (`_provider_cost_from_response`)
- Test: `tests/unit/web/composer/test_llm_response_parsing_bedrock.py` (new)

**Interfaces:**
- Consumes: `token_usage_from_response`, `_provider_cost_from_response` (llm_response_parsing.py:101, 181). Fixture mirror target: `TestProviderCacheTokenAudit._response_with_usage` / its local `FakeResponseWithUsage` dataclass (`tests/unit/web/composer/test_service.py:1974-1991`) — **not** `_make_llm_response`/`FakeChoice` alone (`_helpers.py:142-219`): `FakeLLMResponse` only carries `.choices`, with no `.usage`/`.model`/`._hidden_params`. The new test file defines its own local dataclass following the same pattern (built from `_make_llm_response(...).choices`), extended with `_hidden_params: dict[str, Any]` — a field with no existing fixture precedent.
- Produces: `PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST: ComposerLLMProviderCostSource = "hidden_params.response_cost"`.

```python
ComposerLLMProviderCostSource = Literal["not_available", "response_usage.cost", "hidden_params.response_cost"]
PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST: ComposerLLMProviderCostSource = "hidden_params.response_cost"
_VALID_PROVIDER_COST_SOURCES = {PROVIDER_COST_SOURCE_NOT_AVAILABLE, PROVIDER_COST_SOURCE_RESPONSE_USAGE_COST, PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST}
```
In `_provider_cost_from_response`, when `usage.cost` is absent/invalid, fall back to `getattr(response, "_hidden_params", None)`'s `"response_cost"` key — LiteLLM's `ResponseMetadataHandler.set_hidden_params` (`litellm/litellm_core_utils/llm_response_utils/response_metadata.py:36-56`) computes and attaches this for every provider's calculated cost on every ordinary completion call, including Bedrock — same finite/non-negative guard, source `PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST`.

**Steps:**
- [ ] Write `test_llm_response_parsing_bedrock.py`: define a local `FakeResponseWithHiddenParams` dataclass (mirrors `test_service.py:1984`'s `FakeResponseWithUsage` — `choices: list[FakeChoice]`, `usage: dict[str, Any]`, `model: str`, `id: str` — plus a new `_hidden_params: dict[str, Any] = field(default_factory=dict)`), built from `_make_llm_response(content="Done.").choices`. Build a fake response (`model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"`, `usage={"prompt_tokens": 8200, "completion_tokens": 120, "cache_creation_input_tokens": 7000, "cache_read_input_tokens": 1100}` — no `.cost` key — plus `_hidden_params={"response_cost": 0.01234}`). `test_bedrock_token_and_cache_fields_parse` (`usage.prompt_tokens == 8200`, `usage.completion_tokens == 120`, plus cache-sibling dedup — pins existing correct behavior), `test_bedrock_cost_reads_hidden_params_response_cost` (`_provider_cost_from_response(response) == (0.01234, PROVIDER_COST_SOURCE_HIDDEN_PARAMS_RESPONSE_COST)`), `test_bedrock_hidden_params_cost_non_finite_is_unavailable` (`response_cost=float("nan")` → `(None, PROVIDER_COST_SOURCE_NOT_AVAILABLE)`), `test_openrouter_usage_cost_still_wins_over_hidden_params` (both `usage={"cost": 0.002}` and `_hidden_params={"response_cost": 0.999}` present → `(0.002, PROVIDER_COST_SOURCE_RESPONSE_USAGE_COST)`, pinning that the existing OpenRouter branch runs first).
- [ ] Run `pytest tests/unit/web/composer/test_llm_response_parsing_bedrock.py -v` → the two new cost tests fail: `(None, 'not_available') != (0.01234, 'hidden_params.response_cost')`.
- [ ] Add the literal/constant and the `_hidden_params` fallback branch, checked only after `usage.cost` is absent/invalid.
- [ ] Run the same command → all pass. Run `pytest tests/unit/web/composer/test_service.py -k ProviderCacheTokenAudit -v` → still passes. (This is an unrelated-path sanity check, not a regression guard for this change — every test in that class exercises `token_usage_from_response`, which Task 2 doesn't touch; the actual regression guard for the cost-parsing change is `test_openrouter_usage_cost_still_wins_over_hidden_params` above.)
- [ ] `git add src/elspeth/contracts/composer_llm_audit.py src/elspeth/web/composer/llm_response_parsing.py tests/unit/web/composer/test_llm_response_parsing_bedrock.py && git commit -m "fix(composer): read LiteLLM hidden_params.response_cost for non-OpenRouter providers"`

### Task 3: Bedrock provider-error redaction pin + opt-in live smoke test

**Files:**
- Modify: `tests/unit/web/composer/test_service.py` (new test near line 2989)
- Create: `tests/integration/web/composer/test_bedrock_live_smoke.py`

**Interfaces:**
- Consumes: `ComposerServiceImpl.compose`, `_litellm_acompletion` (service.py:326), `litellm.exceptions.BadRequestError`; `_make_settings`/`_mock_catalog`/`_empty_state` (`_helpers.py`); `integration` marker (pyproject.toml:428, not deselected by default `addopts` :424).

**Steps:**
- [ ] Write `test_bedrock_bad_request_raises_redacted_service_error` in `test_service.py`, mirroring `test_litellm_bad_request_raises_redacted_service_error` (test_service.py:2989-3014) with `model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"`, `llm_provider="bedrock"`, `message="bedrock leaked-detail"` — assert `str(exc_info.value) == "LLM request rejected (BadRequestError)"` and `"leaked-detail" not in str(exc_info.value)`.
- [ ] Run `pytest tests/unit/web/composer/test_service.py -k bedrock -v` → passes immediately: redaction keys on exception type, not provider name, so this pins correct behavior against regression rather than fixing a bug.
- [ ] Write `test_bedrock_live_smoke.py`: `pytestmark = [pytest.mark.integration, pytest.mark.asyncio]`; `_env_present()` checks `ELSPETH_BEDROCK_LIVE_TEST_MODEL`, `AWS_REGION`, and one of `AWS_ACCESS_KEY_ID`/`AWS_PROFILE`/`AWS_CONTAINER_CREDENTIALS_RELATIVE_URI`; `@pytest.mark.skipif(not _env_present(), reason="requires ELSPETH_BEDROCK_LIVE_TEST_MODEL, AWS_REGION, and AWS credentials")`; `test_bedrock_live_completion` awaits the composer's own `_litellm_acompletion` (service.py:326 — not raw `litellm.completion`, so it exercises `ComposerServiceImpl`'s actual call path) with `model=os.environ["ELSPETH_BEDROCK_LIVE_TEST_MODEL"]`, a one-line user message, `aws_region_name=os.environ["AWS_REGION"]`, `max_tokens=16`; asserts `response.choices[0].message.content.strip()` non-empty.
- [ ] Run `pytest tests/integration/web/composer/test_bedrock_live_smoke.py -v` (no env set) → 1 skipped.
- [ ] `git add tests/unit/web/composer/test_service.py tests/integration/web/composer/test_bedrock_live_smoke.py && git commit -m "test(composer): pin Bedrock error redaction and add opt-in live Bedrock smoke test"`

**Notes (verified, deliberately deferred — not blocking this plan):**
- The live-smoke gate is convention-based (env `skipif`), not CI-marker-enforced. Verified sound today: CI's integration stage (`ci.yaml:519-571`) runs `-m integration` unconditionally but its `env:` block (`:554-555`) sets only `OPENROUTER_API_KEY`, never an AWS variable or `ELSPETH_BEDROCK_LIVE_TEST_MODEL`, so the new test self-skips in CI. A dedicated `bedrock_live` marker excluded by default would harden this against a future unrelated AWS-secrets addition (e.g. for `aws_s3`/LocalStack integration tests) — out of scope for this plan.
- `_LLM_ERROR_SENSITIVE_PATTERNS` (`llm_response_parsing.py:65-76`) has no AWS ARN/account-ID pattern, and this is the first plan routing AWS-shaped errors (e.g. botocore `AccessDeniedException` text, which routinely echoes the calling principal's IAM ARN) through composer redaction. ARNs/account IDs are identifiers, not credentials, so this is not a secret-leak risk on the level of the existing patterns — deferred rather than added without a dedicated failing-test-first step of its own.

