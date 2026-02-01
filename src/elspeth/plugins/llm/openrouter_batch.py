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

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from pydantic import Field

from elspeth.contracts import CallStatus, CallType, Determinism, TransformResult
from elspeth.contracts.schema import SchemaConfig
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.pooling import is_capacity_error
from elspeth.plugins.schema_factory import create_schema_from_config


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
                fields: dynamic
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

        # Store OpenRouter-specific settings
        self._api_key = cfg.api_key
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
        self._on_error = cfg.on_error

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
            is_dynamic=schema_config.is_dynamic,
            guaranteed_fields=tuple(set(base_guaranteed) | set(guaranteed)),
            audit_fields=tuple(set(base_audit) | set(audit)),
            required_fields=schema_config.required_fields,
        )

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch of rows in parallel.

        When is_batch_aware=True, the engine passes list[dict].
        For single-row fallback, the engine passes dict.

        Args:
            row: Single row dict OR list of row dicts (batch)
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
        row: dict[str, Any],
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
            # Propagate success_reason from batch result
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
        rows: list[dict[str, Any]],
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

        # Process rows in parallel using a SHARED httpx.Client
        # httpx.Client is thread-safe and reusing it avoids connection overhead
        results: dict[int, dict[str, Any] | Exception] = {}

        with (
            httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "HTTP-Referer": "https://github.com/elspeth-rapid",
                },
            ) as client,
            ThreadPoolExecutor(max_workers=self._pool_size) as executor,
        ):
            # Submit all rows with shared client
            futures = {executor.submit(self._process_single_row, idx, row, ctx, client): idx for idx, row in enumerate(rows)}

            # Collect results - catch only transport exceptions, let plugin bugs crash
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as e:
                    # Transport-level errors are row-level failures, not plugin bugs
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
                # Unexpected exception (httpx transport errors caught in as_completed loop)
                output_row = dict(rows[idx])
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "unexpected_exception",
                    "error": str(result),
                    "error_type": type(result).__name__,
                }
                output_rows.append(output_row)

            elif "error" in result:
                # Row-level error from _process_single_row
                output_row = dict(rows[idx])
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = result["error"]
                output_rows.append(output_row)

            else:
                # Success
                output_rows.append(result)

        return TransformResult.success_multi(
            output_rows,
            success_reason={"action": "enriched", "fields_added": [self._response_field]},
        )

    def _process_single_row(
        self,
        idx: int,
        row: dict[str, Any],
        ctx: PluginContext,
        client: httpx.Client,
    ) -> dict[str, Any]:
        """Process a single row through OpenRouter API.

        Called by worker threads. Returns either the processed row dict
        or a dict with an "error" key.

        Args:
            idx: Row index in batch
            row: Row to process
            ctx: Plugin context
            client: Shared httpx.Client (thread-safe) for making requests

        Returns:
            Processed row dict or {"error": {...}} on failure
        """
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            # Record template error to audit trail - every decision must be traceable
            # per CLAUDE.md auditability standard
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

        # 3. Make API call (EXTERNAL BOUNDARY - wrap and audit)
        # We need state_id to record the call - batch transforms should always have it
        # (synthetic IDs would fail foreign key constraints in calls table)
        state_id = ctx.state_id
        if state_id is None:
            return {"error": {"reason": "missing_state_id"}}

        start = time.perf_counter()
        try:
            # Use the shared httpx.Client passed from _process_batch
            # (httpx.Client is thread-safe, avoids per-row connection overhead)
            response = client.post(
                "/chat/completions",
                json=request_body,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            latency_ms = (time.perf_counter() - start) * 1000

        except httpx.HTTPStatusError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data={"row_index": idx, **request_body},
                response_data=None,
                error={"status_code": e.response.status_code, "error": str(e)},
                latency_ms=latency_ms,
                provider="openrouter",
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
            latency_ms = (time.perf_counter() - start) * 1000
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data={"row_index": idx, **request_body},
                response_data=None,
                error={"error": str(e), "error_type": type(e).__name__},
                latency_ms=latency_ms,
                provider="openrouter",
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
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data={"row_index": idx, **request_body},
                response_data=None,
                error={"reason": "invalid_json", "error": str(e)},
                latency_ms=latency_ms,
                provider="openrouter",
            )
            return {
                "error": {
                    "reason": "invalid_json_response",
                    "error": str(e),
                    **({"body_preview": response.text[:500]} if response.text else {}),
                }
            }

        # 5. Validate response structure (EXTERNAL DATA - validate at boundary)
        if not isinstance(data, dict):
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data={"row_index": idx, **request_body},
                response_data=None,
                error={"reason": "invalid_json_type", "actual": type(data).__name__},
                latency_ms=latency_ms,
                provider="openrouter",
            )
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
                ctx.record_call(
                    call_type=CallType.LLM,
                    status=CallStatus.ERROR,
                    request_data={"row_index": idx, **request_body},
                    response_data=data,
                    error={"reason": "empty_choices"},
                    latency_ms=latency_ms,
                    provider="openrouter",
                )
                return {"error": {"reason": "empty_choices", "response": data}}

            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            ctx.record_call(
                call_type=CallType.LLM,
                status=CallStatus.ERROR,
                request_data={"row_index": idx, **request_body},
                response_data=data,
                error={"reason": "malformed_response", "error": str(e)},
                latency_ms=latency_ms,
                provider="openrouter",
            )
            return {
                "error": {
                    "reason": "malformed_response",
                    "error": f"{type(e).__name__}: {e}",
                    "response_keys": list(data.keys()) if isinstance(data, dict) else None,
                }
            }

        # Record successful call
        # Note: "usage" and "model" are optional in OpenAI/OpenRouter API responses
        # (e.g., streaming responses may omit usage). The .get() here handles a valid
        # API variation, not a bug - this is Tier 3 external data normalization.
        usage = data.get("usage") or {}
        response_model = data.get("model", self._model)
        ctx.record_call(
            call_type=CallType.LLM,
            status=CallStatus.SUCCESS,
            request_data={"row_index": idx, **request_body},
            response_data={"content": content, "usage": usage, "model": response_model},
            latency_ms=latency_ms,
            provider="openrouter",
        )

        # 7. Build output row (OUR CODE - let exceptions crash)
        output = dict(row)
        output[self._response_field] = content
        output[f"{self._response_field}_usage"] = usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        # Template source metadata - always present for audit (None = inline template)
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source
        output[f"{self._response_field}_model"] = response_model

        return output

    def close(self) -> None:
        """Release resources."""
        pass  # No persistent resources to clean up
