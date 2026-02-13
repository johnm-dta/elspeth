# src/elspeth/plugins/llm/openrouter_multi_query.py
"""OpenRouter Multi-Query LLM transform for case study x criteria evaluation.

Executes multiple LLM queries per row via OpenRouter's HTTP API,
merging all results into a single output row with all-or-nothing error handling.

Uses HTTP-based communication (AuditedHTTPClient) rather than SDK-based
communication like the Azure variant.

Inherits row-level pipelining (BatchTransformMixin) and query-level concurrency
(PooledExecutor) from BaseMultiQueryTransform.
"""

from __future__ import annotations

import json
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx

from elspeth.contracts import TransformResult
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.plugins.llm.base_multi_query import BaseMultiQueryTransform
from elspeth.plugins.llm.multi_query import (
    MultiQueryConfigMixin,
    QuerySpec,
    ResponseFormat,
)
from elspeth.plugins.llm.openrouter import OpenRouterConfig
from elspeth.plugins.llm.templates import TemplateError
from elspeth.plugins.llm.tracing import (
    LangfuseTracingConfig,
    validate_tracing_config,
)
from elspeth.plugins.pooling import CapacityError, is_capacity_error

if TYPE_CHECKING:
    from elspeth.contracts import TransformErrorReason


class OpenRouterMultiQueryConfig(OpenRouterConfig, MultiQueryConfigMixin):
    """Configuration for OpenRouter multi-query LLM transform.

    Combines OpenRouterConfig (HTTP connection settings, pooling, templates)
    with MultiQueryConfigMixin (case_studies, criteria, output_mapping).

    The cross-product of case_studies x criteria defines all queries.
    """


# Resolve forward references for Pydantic (CaseStudyConfig, CriterionConfig)
OpenRouterMultiQueryConfig.model_rebuild()


class OpenRouterMultiQueryLLMTransform(BaseMultiQueryTransform):
    """LLM transform that executes case_studies x criteria queries per row via OpenRouter.

    For each row, expands the cross-product of case studies and criteria
    into individual LLM queries. All queries run in parallel (up to pool_size),
    with all-or-nothing error semantics (if any query fails, the row fails).

    Uses HTTP-based communication via AuditedHTTPClient, unlike the Azure
    variant which uses the OpenAI SDK.

    Configuration example:
        transforms:
          - plugin: openrouter_multi_query_llm
            options:
              model: "anthropic/claude-3-opus"
              api_key: "${OPENROUTER_API_KEY}"
              template: |
                Case: {{ input_1 }}, {{ input_2 }}
                Criterion: {{ criterion.name }}
              case_studies:
                - name: cs1
                  input_fields: [cs1_bg, cs1_sym]
              criteria:
                - name: diagnosis
                  code: DIAG
              response_format: structured
              output_mapping:
                score:
                  suffix: score
                  type: integer
              pool_size: 4
              schema:
                mode: observed
    """

    name = "openrouter_multi_query_llm"

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize transform with multi-query configuration."""
        super().__init__(config)

        cfg = OpenRouterMultiQueryConfig.from_dict(config)

        # Pre-build auth headers — avoids storing the raw API key as a named attribute
        self._request_headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "HTTP-Referer": "https://github.com/elspeth-rapid",  # Required by OpenRouter
        }
        self._base_url = cfg.base_url
        self._timeout = cfg.timeout_seconds
        self._model = cfg.model

        # Shared multi-query init (template, schema, executor, tracing)
        self._init_multi_query(cfg)

        # HTTP client caching (thread-safe)
        self._http_clients: dict[str, AuditedHTTPClient] = {}
        self._http_clients_lock = Lock()

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_rate_limiter_service_name(self) -> str:
        return "openrouter"

    def _cleanup_clients(self, state_id: str) -> None:
        with self._http_clients_lock:
            client = self._http_clients.pop(state_id, None)
        if client is not None:
            client.close()

    def _close_all_clients(self) -> None:
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()

    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider.

        OpenRouter uses HTTP directly (not the OpenAI SDK), so Azure AI
        auto-instrumentation is NOT supported. Only Langfuse (manual spans)
        is available.
        """
        import structlog

        logger = structlog.get_logger(__name__)

        assert self._tracing_config is not None
        tracing_config = self._tracing_config

        errors = validate_tracing_config(tracing_config)
        if errors:
            for error in errors:
                logger.warning("Tracing configuration error", error=error)
            return

        match tracing_config.provider:
            case "azure_ai":
                logger.warning(
                    "Azure AI tracing not supported for OpenRouter - use Langfuse instead",
                    provider="azure_ai",
                    hint="Azure AI auto-instruments the OpenAI SDK; OpenRouter uses HTTP directly",
                )
            case "langfuse":
                self._setup_langfuse_tracing(logger)
            case "none":
                pass
            case _:
                logger.warning(
                    "Unknown tracing provider encountered after validation - tracing disabled",
                    provider=tracing_config.provider,
                )

    def _process_single_query(
        self,
        row: PipelineRow | dict[str, Any],
        spec: QuerySpec,
        state_id: str,
        token_id: str,
        input_contract: SchemaContract | None,
    ) -> TransformResult:
        """Process a single query (one case_study x criterion pair) via HTTP.

        Args:
            row: Full input row
            spec: Query specification with input field mapping
            state_id: State ID for audit trail
            token_id: Token ID for tracing correlation
            input_contract: Schema contract for template dual-name access

        Returns:
            TransformResult with mapped output fields

        Raises:
            CapacityError: On rate limit (429/503/529) for pooled retry
        """
        # 1. Build synthetic row for PromptTemplate
        synthetic_row = spec.build_template_context(row)

        # 2. Render template (THEIR DATA - wrap in try/catch)
        # BUG FIX (d9yk/fd40): Pass contract for dual-name template access
        try:
            rendered = self._template.render_with_metadata(synthetic_row, contract=input_contract)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "query": spec.output_prefix,
                    "template_hash": self._template.template_hash,
                }
            )

        # 3. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 4. Build HTTP request body
        effective_max_tokens = spec.max_tokens or self._max_tokens

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
            "response_format": self._response_format_dict,
        }
        if effective_max_tokens:
            request_body["max_tokens"] = effective_max_tokens

        # 5. Get HTTP client
        http_client = self._get_http_client(state_id, token_id=token_id)

        # 6. Call OpenRouter API (EXTERNAL - wrap, raise CapacityError for retry)
        try:
            response = http_client.post(
                "/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if is_capacity_error(status_code):
                raise CapacityError(status_code, str(e)) from e
            return TransformResult.error(
                {
                    "reason": "api_call_failed",
                    "error": str(e),
                    "status_code": status_code,
                    "query": spec.output_prefix,
                },
                retryable=status_code >= 500,
            )
        except httpx.RequestError as e:
            return TransformResult.error(
                {
                    "reason": "api_call_failed",
                    "error": str(e),
                    "query": spec.output_prefix,
                },
                retryable=True,
            )

        # 7. Parse JSON response from HTTP (EXTERNAL DATA - wrap)
        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            json_error: TransformErrorReason = {
                "reason": "invalid_json_response",
                "error": f"Response is not valid JSON: {e}",
                "query": spec.output_prefix,
                "content_type": response.headers.get("content-type", "unknown"),
            }
            if response.text:
                json_error["body_preview"] = response.text[:500]
            return TransformResult.error(json_error, retryable=False)

        # 8. Extract content from OpenRouter response (EXTERNAL DATA - wrap)
        try:
            choices = data["choices"]
            if not choices:
                return TransformResult.error(
                    {
                        "reason": "empty_choices",
                        "query": spec.output_prefix,
                        "response": data,
                    },
                    retryable=False,
                )
            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            return TransformResult.error(
                {
                    "reason": "malformed_response",
                    "error": f"{type(e).__name__}: {e}",
                    "query": spec.output_prefix,
                    "response_keys": list(data.keys()) if isinstance(data, dict) else None,
                },
                retryable=False,
            )

        # 8c. Check for content filtering (null content from provider)
        if content is None:
            return TransformResult.error(
                {
                    "reason": "content_filtered",
                    "error": "LLM returned null content (likely content-filtered by provider)",
                    "query": spec.output_prefix,
                },
                retryable=False,
            )

        # OpenRouter can return {"usage": null} or omit usage entirely.
        # dict.get("usage", {}) only returns {} when key is MISSING, not when value is null.
        # The `or {}` ensures we get an empty dict for both missing AND null cases.
        usage = data.get("usage") or {}

        # 8b. Check for response truncation BEFORE parsing
        # usage is Tier 3 external data - use .get() for optional fields
        completion_tokens = usage.get("completion_tokens", 0)
        if effective_max_tokens is not None and completion_tokens >= effective_max_tokens:
            truncation_error: TransformErrorReason = {
                "reason": "response_truncated",
                "error": (
                    f"LLM response was truncated at {completion_tokens} tokens "
                    f"(max_tokens={effective_max_tokens}). "
                    f"Increase max_tokens for query '{spec.output_prefix}' or shorten your prompt."
                ),
                "query": spec.output_prefix,
                "max_tokens": effective_max_tokens,
                "completion_tokens": completion_tokens,
                "prompt_tokens": usage.get("prompt_tokens", 0),
            }
            if content:
                truncation_error["raw_response_preview"] = content[:500]
            return TransformResult.error(truncation_error)

        # 9. Parse LLM response content as JSON (THEIR DATA - wrap)
        content_str = content.strip()

        # Strip markdown code blocks if present (common in standard mode, not in structured mode)
        if self._response_format == ResponseFormat.STANDARD and content_str.startswith("```"):
            first_newline = content_str.find("\n")
            if first_newline != -1:
                content_str = content_str[first_newline + 1 :]
            if content_str.endswith("```"):
                content_str = content_str[:-3].strip()

        try:
            parsed = json.loads(content_str)
        except json.JSONDecodeError as e:
            parse_error: TransformErrorReason = {
                "reason": "json_parse_failed",
                "error": str(e),
                "query": spec.output_prefix,
            }
            if content:
                parse_error["raw_response"] = content[:500]
            return TransformResult.error(parse_error)

        # Validate JSON type is object (EXTERNAL DATA - validate structure)
        if not isinstance(parsed, dict):
            json_type_error: TransformErrorReason = {
                "reason": "invalid_json_type",
                "expected": "object",
                "actual": type(parsed).__name__,
                "query": spec.output_prefix,
            }
            if content:
                json_type_error["raw_response"] = content[:500]
            return TransformResult.error(json_type_error)

        # 10. Map and validate output fields
        output: dict[str, Any] = {}
        for json_field, field_config in self._output_mapping.items():
            output_key = f"{spec.output_prefix}_{field_config.suffix}"
            if json_field not in parsed:
                return TransformResult.error(
                    {
                        "reason": "missing_output_field",
                        "field": json_field,
                        "query": spec.output_prefix,
                    }
                )

            value = parsed[json_field]

            type_error = self._validate_field_type(json_field, value, field_config)
            if type_error is not None:
                return TransformResult.error(
                    {
                        "reason": "type_mismatch",
                        "field": json_field,
                        "expected": field_config.type.value,
                        "actual": type(value).__name__,
                        "value": str(value)[:100],
                        "query": spec.output_prefix,
                    }
                )

            output[output_key] = value

        # 11. Add metadata for audit trail
        output[f"{spec.output_prefix}_usage"] = usage
        output[f"{spec.output_prefix}_model"] = data.get("model", self._model)
        output[f"{spec.output_prefix}_template_hash"] = rendered.template_hash
        output[f"{spec.output_prefix}_variables_hash"] = rendered.variables_hash
        output[f"{spec.output_prefix}_template_source"] = rendered.template_source
        output[f"{spec.output_prefix}_lookup_hash"] = rendered.lookup_hash
        output[f"{spec.output_prefix}_lookup_source"] = rendered.lookup_source
        output[f"{spec.output_prefix}_system_prompt_source"] = self._system_prompt_source

        fields_added = [f"{spec.output_prefix}_{field_config.suffix}" for field_config in self._output_mapping.values()]
        observed = SchemaContract(
            mode="OBSERVED",
            fields=tuple(FieldContract(k, k, object, False, "inferred") for k in output),
            locked=True,
        )
        return TransformResult.success(
            PipelineRow(output, observed),
            success_reason={"action": "enriched", "fields_added": fields_added},
        )

    # ------------------------------------------------------------------
    # OpenRouter-specific client management
    # ------------------------------------------------------------------

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> AuditedHTTPClient:
        """Get or create HTTP client for a state_id.

        Clients are cached to preserve call_index across retries.
        This ensures uniqueness of (state_id, call_index) even when
        the pooled executor retries after CapacityError.

        Thread-safe: multiple workers can call this concurrently.
        """
        with self._http_clients_lock:
            if state_id not in self._http_clients:
                if self._recorder is None:
                    raise RuntimeError("OpenRouter multi-query transform requires recorder. Ensure on_start was called.")
                self._http_clients[state_id] = AuditedHTTPClient(
                    recorder=self._recorder,
                    state_id=state_id,
                    run_id=self._run_id,
                    telemetry_emit=self._telemetry_emit,
                    timeout=self._timeout,
                    base_url=self._base_url,
                    headers=self._request_headers,
                    limiter=self._limiter,
                    token_id=token_id,
                )
            return self._http_clients[state_id]

    # ------------------------------------------------------------------
    # OpenRouter-specific tracing
    # ------------------------------------------------------------------

    def _setup_langfuse_tracing(self, logger: Any) -> None:
        """Initialize Langfuse tracing (v3 API).

        Langfuse v3 uses OpenTelemetry-based context managers for lifecycle.
        The Langfuse client is stored for use in _record_row_langfuse_trace().
        """
        try:
            from langfuse import Langfuse  # type: ignore[import-not-found,import-untyped]  # optional dep, no stubs

            cfg = self._tracing_config
            if not isinstance(cfg, LangfuseTracingConfig):
                return

            self._langfuse_client = Langfuse(
                public_key=cfg.public_key,
                secret_key=cfg.secret_key,
                host=cfg.host,
                tracing_enabled=cfg.tracing_enabled,
            )
            self._tracing_active = True

            logger.info(
                "Langfuse tracing initialized (v3)",
                provider="langfuse",
                host=cfg.host,
                tracing_enabled=cfg.tracing_enabled,
            )

        except ImportError:
            logger.warning(
                "Langfuse tracing requested but package not installed",
                provider="langfuse",
                hint="Install with: uv pip install elspeth[tracing-langfuse]",
            )
