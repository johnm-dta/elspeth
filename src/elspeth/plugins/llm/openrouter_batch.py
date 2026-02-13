# src/elspeth/plugins/llm/openrouter_batch.py
"""OpenRouter batch LLM transform for aggregation pipelines.

Batch-aware transform that processes multiple rows in parallel via OpenRouter API.
Unlike azure_batch_llm (which uses Azure's async batch API with checkpointing),
this plugin processes rows synchronously in parallel using concurrent HTTP requests.

Use this plugin in aggregation nodes where the engine buffers rows until a trigger
fires, then calls process() with the entire batch.

Benefits:
- Parallel processing of buffered rows
- Simple synchronous model (no checkpointing needed)
- Works with any OpenRouter-supported model
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field

from elspeth.contracts import CallStatus, CallType, Determinism, TransformResult
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.http import AuditedHTTPClient
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.llm.tracing import (
    LangfuseTracingConfig,
    TracingConfig,
    parse_tracing_config,
    validate_tracing_config,
)
from elspeth.plugins.pooling import is_capacity_error
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from elspeth.core.landscape.recorder import LandscapeRecorder


class OpenRouterBatchConfig(LLMConfig):
    """OpenRouter batch-specific configuration.

    Extends LLMConfig with OpenRouter API settings and batch processing options.

    Required fields:
        api_key: OpenRouter API key
        model: Model identifier (e.g., "openai/gpt-4o-mini")
        template: Jinja2 prompt template

    Optional fields:
        base_url: API base URL (default: https://openrouter.ai/api/v1)
        timeout_seconds: Request timeout (default: 60.0)
        pool_size: Number of parallel workers (default: 5)
        system_prompt: Optional system message
        temperature: Sampling temperature (default: 0.0)
        max_tokens: Maximum response tokens
        response_field: Output field name (default: llm_response)
    """

    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    timeout_seconds: float = Field(default=60.0, gt=0, description="Request timeout")

    # Tier 2: Plugin-internal tracing (optional, Langfuse only)
    # Azure AI tracing is NOT supported - it auto-instruments the OpenAI SDK,
    # but OpenRouter uses HTTP directly via httpx.
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (langfuse only - azure_ai not supported)",
    )


class OpenRouterBatchLLMTransform(BaseTransform):
    """Batch-aware LLM transform using OpenRouter API.

    Processes batches of rows in parallel via concurrent HTTP requests to OpenRouter.
    Designed for use in aggregation nodes where the engine buffers rows until a
    trigger fires.

    Unlike azure_batch_llm which uses Azure's async batch API:
    - No checkpointing needed (synchronous processing)
    - Parallel HTTP requests within a single process() call
    - Immediate results (no polling)

    Architecture:
        Engine buffers rows → trigger fires → process(rows: list[dict]) called
        → ThreadPoolExecutor makes parallel HTTP calls → results assembled
        → TransformResult.success_multi(output_rows)

    Configuration example:
        aggregations:
          - name: sentiment_batch
            plugin: openrouter_batch_llm
            trigger:
              count: 10  # Process every 10 rows
            output_mode: passthrough
            options:
              api_key: "${OPENROUTER_API_KEY}"
              model: "openai/gpt-4o-mini"
              template: |
                Analyze: {{ row.text }}
              pool_size: 5  # Parallel workers
              schema:
                mode: observed
    """

    name = "openrouter_batch_llm"
    is_batch_aware = True  # Engine passes list[dict] for batch processing

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize OpenRouter batch LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        # Parse OpenRouter-specific config
        cfg = OpenRouterBatchConfig.from_dict(config)

        # Pre-build auth headers — avoids storing the raw API key as a named attribute
        self._request_headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "HTTP-Referer": "https://github.com/elspeth-rapid",
        }
        self._base_url = cfg.base_url
        self._timeout = cfg.timeout_seconds

        # Store common LLM settings
        # Note: max_capacity_retry_seconds not stored - this plugin uses simple
        # ThreadPoolExecutor, not PooledExecutor with AIMD retry. Row-level retries
        # are handled by the engine's RetryManager based on retryable error flags.
        self._pool_size = cfg.pool_size
        self._model = cfg.model
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._system_prompt_source = cfg.system_prompt_source
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field

        # Schema from config (TransformDataConfig guarantees schema_config is not None)
        schema_config = cfg.schema_config
        schema = create_schema_from_config(
            schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

        # Build output schema config with field categorization
        guaranteed = get_llm_guaranteed_fields(self._response_field)
        audit = get_llm_audit_fields(self._response_field)

        # Merge with any existing fields from base schema
        base_guaranteed = schema_config.guaranteed_fields or ()
        base_audit = schema_config.audit_fields or ()

        self._output_schema_config = SchemaConfig(
            mode=schema_config.mode,
            fields=schema_config.fields,
            guaranteed_fields=tuple(set(base_guaranteed) | set(guaranteed)),
            audit_fields=tuple(set(base_audit) | set(audit)),
            required_fields=schema_config.required_fields,
        )

        # Recorder and telemetry references (set in on_start)
        self._recorder: LandscapeRecorder | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = lambda event: None
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        # HTTP client cache — one per state_id for call_index uniqueness.
        # Each state_id gets its own AuditedHTTPClient with monotonically
        # increasing call indices, ensuring UNIQUE(state_id, call_index).
        self._http_clients: dict[str, AuditedHTTPClient] = {}
        self._http_clients_lock = Lock()

        # Tier 2: Plugin-internal tracing (Langfuse only)
        self._tracing_config: TracingConfig | None = parse_tracing_config(cfg.tracing)
        self._tracing_active: bool = False
        self._langfuse_client: Any = None  # Langfuse client if configured

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder, telemetry, rate limit context, and initialize tracing.

        Called by the engine at pipeline start. Captures the landscape
        recorder, run_id, telemetry callback, and rate limiter for use in
        worker threads. Also initializes Tier 2 tracing if configured.
        """
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        self._limiter = ctx.rate_limit_registry.get_limiter("openrouter") if ctx.rate_limit_registry is not None else None

        if self._tracing_config is not None:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider.

        OpenRouter uses HTTP directly (not the OpenAI SDK), so Azure AI
        auto-instrumentation is NOT supported. Only Langfuse (manual spans)
        is available.
        """
        import structlog

        logger = structlog.get_logger(__name__)

        if self._tracing_config is None:
            return

        # Validate configuration completeness
        errors = validate_tracing_config(self._tracing_config)
        if errors:
            for error in errors:
                logger.warning("Tracing configuration error", error=error)
            return

        match self._tracing_config.provider:
            case "azure_ai":
                # Azure AI tracing NOT supported for OpenRouter
                logger.warning(
                    "Azure AI tracing not supported for OpenRouter - use Langfuse instead",
                    provider="azure_ai",
                    hint="Azure AI auto-instruments the OpenAI SDK; OpenRouter uses HTTP directly",
                )
                return
            case "langfuse":
                self._setup_langfuse_tracing(logger)
            case "none":
                pass  # No tracing
            case _:
                logger.warning(
                    "Unknown tracing provider encountered after validation - tracing disabled",
                    provider=self._tracing_config.provider,
                )

    def _setup_langfuse_tracing(self, logger: Any) -> None:
        """Initialize Langfuse tracing (v3 API).

        Langfuse v3 uses OpenTelemetry-based context managers for lifecycle.
        The Langfuse client is stored for use in _record_langfuse_trace().
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

    def _record_langfuse_trace(
        self,
        idx: int,
        prompt: str,
        response_content: str,
        model: str,
        usage: dict[str, int] | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        """Record LLM call to Langfuse using v3 nested context managers.

        Unlike Azure Batch, OpenRouter batch processes rows via synchronous
        HTTP calls in a ThreadPoolExecutor, so we CAN trace each call.

        Args:
            idx: Row index in the batch
            prompt: The prompt sent to the LLM
            response_content: The response received (empty if error)
            model: Model name
            usage: Token usage dict with prompt_tokens/completion_tokens
            latency_ms: Call latency in milliseconds
            error: Error message if call failed
        """
        if not self._tracing_active or self._langfuse_client is None:
            return

        try:
            with (
                self._langfuse_client.start_as_current_observation(
                    as_type="span",
                    name=f"elspeth.{self.name}",
                    metadata={
                        "row_index": idx,
                        "plugin": self.name,
                        "model": model,
                    },
                ),
                self._langfuse_client.start_as_current_observation(
                    as_type="generation",
                    name="llm_call",
                    model=model,
                    input=[{"role": "user", "content": prompt}],
                ) as generation,
            ):
                update_kwargs: dict[str, Any] = {
                    "output": response_content if not error else None,
                }

                if usage:
                    # Validate types at external boundary (Tier 3 data from LLM API)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                        update_kwargs["usage_details"] = {
                            "input": prompt_tokens,
                            "output": completion_tokens,
                        }

                metadata: dict[str, Any] = {"row_index": idx}
                if latency_ms is not None:
                    metadata["latency_ms"] = latency_ms
                if error:
                    metadata["error"] = error
                    update_kwargs["level"] = "ERROR"
                update_kwargs["metadata"] = metadata

                generation.update(**update_kwargs)

        except Exception as e:
            import structlog

            logger = structlog.get_logger(__name__)
            logger.warning("Failed to record Langfuse trace", error=str(e))

    def process(
        self,
        row: PipelineRow | list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch of rows in parallel.

        When is_batch_aware=True and configured as aggregation, the engine passes list[PipelineRow].
        For single-row fallback (non-aggregation), the engine passes PipelineRow.

        Args:
            row: Single PipelineRow OR list[PipelineRow] (batch mode)
            ctx: Plugin context

        Returns:
            TransformResult with processed rows
        """
        if isinstance(row, list):
            return self._process_batch(row, ctx)
        else:
            # Single row fallback - wrap in list and process
            return self._process_single(row, ctx)

    def _process_single(
        self,
        row: PipelineRow,
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row (fallback for non-batch mode).

        Args:
            row: Single row dict
            ctx: Plugin context

        Returns:
            TransformResult with processed row
        """
        result = self._process_batch([row], ctx)

        # Convert multi-row result back to single-row
        if result.status == "success" and result.rows:
            # result.rows[0] is already PipelineRow from _process_batch
            return TransformResult.success(
                result.rows[0],
                success_reason=result.success_reason or {"action": "enriched", "fields_added": [self._response_field]},
            )
        elif result.status == "error":
            return result
        else:
            # Defense-in-depth: _process_batch() should always return either:
            # - success_multi(rows) with non-empty rows
            # - error()
            # If we reach here, something unexpected happened - crash rather than
            # silently passing through the original row unprocessed.
            raise RuntimeError(
                f"Unexpected result from _process_batch: status={result.status}, "
                f"row={result.row}, rows={result.rows}. "
                f"Expected success with rows or error. This indicates a bug in "
                f"_process_batch or an upstream change that broke the contract."
            )

    def _process_batch(
        self,
        rows: list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch of rows in parallel via ThreadPoolExecutor.

        Args:
            rows: List of row dicts to process
            ctx: Plugin context

        Returns:
            TransformResult with all processed rows
        """
        if not rows:
            # Engine invariant: AggregationExecutor.execute_flush() guards against empty buffers.
            # If we reach here, something bypassed that guard - this is a bug, not a valid state.
            # Per CLAUDE.md "crash on plugin bugs" principle, fail fast rather than emit garbage.
            raise RuntimeError(
                f"Empty batch passed to batch-aware transform '{self.name}'. "
                f"This should never happen - AggregationExecutor.execute_flush() guards against "
                f"empty buffers. This indicates a bug in the engine or test setup."
            )

        # Process rows in parallel using AuditedHTTPClient (one per state_id).
        # Each worker thread gets a cached client via _get_http_client().
        results: dict[int, dict[str, Any] | Exception] = {}

        with ThreadPoolExecutor(max_workers=self._pool_size) as executor:
            futures = {executor.submit(self._process_single_row, idx, row, ctx): idx for idx, row in enumerate(rows)}

            # Collect results - catch only transport exceptions, let plugin bugs crash.
            # _process_single_row already handles HTTPStatusError and RequestError
            # (the two HTTPError subclasses). Only StreamError can legitimately
            # escape — it occurs during response streaming, after the request succeeds.
            # InvalidURL is a config bug (bad base_url) and must crash per CLAUDE.md.
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except httpx.StreamError as e:
                    results[idx] = e

        # Assemble output rows in original order
        # Every row gets an output (success or with error markers) - no rows are dropped
        output_rows: list[dict[str, Any]] = []

        for idx in range(len(rows)):
            if idx not in results:
                raise RuntimeError(
                    f"OpenRouter batch results missing for row index {idx}. This indicates an internal concurrency or collection bug."
                )
            result = results[idx]

            if isinstance(result, Exception):
                # StreamError escaped from _process_single_row — record to audit
                # trail so the failure is attributable (h27 fix).
                ctx.record_call(
                    call_type=CallType.LLM,
                    status=CallStatus.ERROR,
                    request_data={"row_index": idx},
                    response_data=None,
                    error={
                        "reason": "transport_exception",
                        "error": str(result),
                        "error_type": type(result).__name__,
                    },
                    latency_ms=None,
                    provider="openrouter",
                )
                output_row = rows[idx].to_dict()
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "transport_exception",
                    "error": str(result),
                    "error_type": type(result).__name__,
                }
                output_rows.append(output_row)

            elif "error" in result:
                # Row-level error from _process_single_row
                output_row = rows[idx].to_dict()
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = result["error"]
                output_rows.append(output_row)

            else:
                # Success
                output_rows.append(result)

        # Create OBSERVED contract from union of ALL output row keys (not just first)
        # Error rows may have extra fields (e.g. _error) that the first row lacks
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        all_keys: dict[str, None] = {}
        for r in output_rows:
            for key in r:
                all_keys[key] = None

        fields = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=object,  # OBSERVED mode - infer all as object type
                required=False,
                source="inferred",
            )
            for key in all_keys
        )
        output_contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        return TransformResult.success_multi(
            [PipelineRow(r, output_contract) for r in output_rows],
            success_reason={"action": "enriched", "fields_added": [self._response_field]},
        )

    def _get_http_client(self, state_id: str, *, token_id: str | None = None) -> AuditedHTTPClient:
        """Get or create AuditedHTTPClient for a state_id.

        Clients are cached to preserve call_index across retries within the
        same batch. This ensures uniqueness of (state_id, call_index).

        Thread-safe: multiple workers can call this concurrently.
        """
        with self._http_clients_lock:
            if state_id not in self._http_clients:
                if self._recorder is None:
                    raise RuntimeError("OpenRouter batch transform requires recorder. Ensure on_start was called.")
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

    def _process_single_row(
        self,
        idx: int,
        row: PipelineRow,
        ctx: PluginContext,
    ) -> dict[str, Any]:
        """Process a single row through OpenRouter API.

        Called by worker threads. Returns either the processed row dict
        or a dict with an "error" key.

        Uses AuditedHTTPClient which automatically records HTTP calls to the
        Landscape audit trail and emits telemetry events. Manual ctx.record_call()
        is only used for pre-HTTP errors (template rendering).

        Args:
            idx: Row index in batch
            row: Row to process
            ctx: Plugin context

        Returns:
            Processed row dict or {"error": {...}} on failure
        """
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row, contract=row.contract)
        except TemplateError as e:
            # Record template error to audit trail — this happens before any HTTP
            # call, so AuditedHTTPClient can't record it. Use ctx.record_call().
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data={
                    "row_index": idx,
                    "stage": "template_rendering",
                    "template_hash": self._template.template_hash,
                },
                response_data=None,
                error={
                    "reason": "template_rendering_failed",
                    "error": str(e),
                },
                latency_ms=None,
                provider="openrouter",
            )
            return {"error": {"reason": "template_rendering_failed", "error": str(e)}}

        # 2. Build request body (OUR CODE - let exceptions crash)
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        request_body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens:
            request_body["max_tokens"] = self._max_tokens

        # 3. Make API call via AuditedHTTPClient (automatically records to audit trail)
        state_id = ctx.state_id
        if state_id is None:
            raise RuntimeError("OpenRouter batch transform requires state_id. Ensure transform is executed through the engine.")

        # Resolve per-row token_id for telemetry attribution. In batch mode,
        # ctx.batch_token_ids maps row index to token_id (set by AggregationExecutor).
        # Falls back to ctx.token for single-row mode or legacy callers.
        if ctx.batch_token_ids is not None and idx < len(ctx.batch_token_ids):
            row_token_id = ctx.batch_token_ids[idx]
        elif ctx.token is not None:
            row_token_id = ctx.token.token_id
        else:
            row_token_id = None

        http_client = self._get_http_client(state_id, token_id=row_token_id)

        try:
            # AuditedHTTPClient.post() records the call to Landscape and emits
            # telemetry automatically. It does NOT call raise_for_status() — we
            # do that ourselves below to handle error responses per-row.
            # Per-call token_id ensures correct telemetry attribution in batches.
            response = http_client.post(
                "/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
                token_id=row_token_id,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            # HTTP error already recorded by AuditedHTTPClient — just trace and return
            self._record_langfuse_trace(
                idx=idx,
                prompt=rendered.prompt,
                response_content="",
                model=self._model,
                error=f"HTTP {e.response.status_code}: {e}",
            )
            return {
                "error": {
                    "reason": "api_call_failed",
                    "error": str(e),
                    "status_code": e.response.status_code,
                    "retryable": is_capacity_error(e.response.status_code),
                }
            }
        except httpx.RequestError as e:
            # Network error already recorded by AuditedHTTPClient — just trace and return
            self._record_langfuse_trace(
                idx=idx,
                prompt=rendered.prompt,
                response_content="",
                model=self._model,
                error=f"Request error: {e}",
            )
            return {
                "error": {
                    "reason": "api_call_failed",
                    "error": str(e),
                    "retryable": False,
                }
            }

        # 4. Parse JSON response (EXTERNAL DATA - wrap)
        try:
            data = response.json()
        except (ValueError, TypeError) as e:
            return {
                "error": {
                    "reason": "invalid_json_response",
                    "error": str(e),
                    **({"body_preview": response.text[:500]} if response.text else {}),
                }
            }

        # 5. Validate response structure (EXTERNAL DATA - validate at boundary)
        if not isinstance(data, dict):
            return {
                "error": {
                    "reason": "invalid_json_type",
                    "expected": "object",
                    "actual": type(data).__name__,
                }
            }

        # 6. Extract content (EXTERNAL DATA - wrap)
        try:
            choices = data["choices"]
            if not choices:
                return {"error": {"reason": "empty_choices", "response": data}}

            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            return {
                "error": {
                    "reason": "malformed_response",
                    "error": f"{type(e).__name__}: {e}",
                    "response_keys": list(data.keys()) if isinstance(data, dict) else None,
                }
            }

        # Note: "usage" and "model" are optional in OpenAI/OpenRouter API responses
        # (e.g., streaming responses may omit usage). The .get() here handles a valid
        # API variation, not a bug — this is Tier 3 external data normalization.
        usage = data.get("usage") or {}
        response_model = data.get("model", self._model)

        # Record to Langfuse (per-call tracing — unlike Azure Batch, we control each call)
        self._record_langfuse_trace(
            idx=idx,
            prompt=rendered.prompt,
            response_content=content,
            model=response_model,
            usage=usage,
        )

        # 7. Build output row (OUR CODE - let exceptions crash)
        output = row.to_dict()
        output[self._response_field] = content
        output[f"{self._response_field}_usage"] = usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source
        output[f"{self._response_field}_model"] = response_model

        return output

    def close(self) -> None:
        """Release resources and flush tracing."""
        # Flush Tier 2 tracing if active
        if self._tracing_active:
            self._flush_tracing()

        # Close and clear cached HTTP clients
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()

        self._recorder = None
        self._langfuse_client = None

    def _flush_tracing(self) -> None:
        """Flush any pending tracing data."""
        import structlog

        logger = structlog.get_logger(__name__)

        # Langfuse needs explicit flush
        if self._langfuse_client is not None:
            try:
                self._langfuse_client.flush()
                logger.debug("Langfuse tracing flushed")
            except Exception as e:
                logger.warning("Failed to flush Langfuse tracing", error=str(e))
