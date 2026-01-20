# src/elspeth/plugins/llm/azure.py
"""Azure OpenAI LLM transform with optional pooled execution.

Self-contained transform that creates its own AuditedLLMClient using
the context's landscape and state_id. Supports both sequential (pool_size=1)
and pooled (pool_size>1) execution modes.
"""

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, Any, Self

from pydantic import Field, model_validator

from elspeth.contracts import Determinism, TransformResult
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.clients.llm import AuditedLLMClient, LLMClientError, RateLimitError
from elspeth.plugins.context import PluginContext
from elspeth.plugins.llm.base import LLMConfig
from elspeth.plugins.llm.capacity_errors import CapacityError
from elspeth.plugins.llm.pooled_executor import PooledExecutor, RowContext
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.schema_factory import create_schema_from_config

if TYPE_CHECKING:
    from openai import AzureOpenAI

    from elspeth.core.landscape.recorder import LandscapeRecorder


class AzureOpenAIConfig(LLMConfig):
    """Azure OpenAI-specific configuration.

    Extends LLMConfig with Azure-specific settings:
    - deployment_name: Azure deployment name (required) - used as model identifier
    - endpoint: Azure OpenAI endpoint URL (required)
    - api_key: Azure OpenAI API key (required)
    - api_version: Azure API version (default: 2024-10-21)

    Pooling options (inherited from LLMConfig):
    - pool_size: Number of concurrent workers (1=sequential, >1=pooled)
    - max_dispatch_delay_ms: Maximum AIMD backoff delay
    - max_capacity_retry_seconds: Timeout for capacity error retries

    Note: The 'model' field from LLMConfig is automatically set to
    deployment_name if not explicitly provided.
    """

    # Override model to make it optional - will default to deployment_name
    model: str = Field(default="", description="Model identifier (defaults to deployment_name)")

    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="Azure API version")

    @model_validator(mode="after")
    def _set_model_from_deployment(self) -> Self:
        """Set model to deployment_name if not explicitly provided."""
        if not self.model:
            self.model = self.deployment_name
        return self


class AzureLLMTransform(BaseTransform):
    """LLM transform using Azure OpenAI with optional pooled execution.

    Self-contained transform that creates its own AuditedLLMClient
    internally using ctx.landscape and ctx.state_id. Supports both
    sequential (pool_size=1) and pooled (pool_size>1) execution.

    Configuration example:
        transforms:
          - plugin: azure_llm
            options:
              deployment_name: "my-gpt4o-deployment"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Analyze: {{ row.text }}
              schema:
                fields: dynamic
              pool_size: 5  # Enable pooled execution
    """

    name = "azure_llm"
    is_batch_aware = True  # Enable aggregation buffering

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Azure LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        # Parse Azure-specific config to validate all required fields
        cfg = AzureOpenAIConfig.from_dict(config)

        # Store Azure-specific config
        self._azure_endpoint = cfg.endpoint
        self._azure_api_key = cfg.api_key
        self._azure_api_version = cfg.api_version
        self._deployment_name = cfg.deployment_name

        # Store common LLM settings (from LLMConfig)
        self._model = cfg.model or cfg.deployment_name
        self._template = PromptTemplate(
            cfg.template,
            template_source=cfg.template_source,
            lookup_data=cfg.lookup,
            lookup_source=cfg.lookup_source,
        )
        self._system_prompt = cfg.system_prompt
        self._temperature = cfg.temperature
        self._max_tokens = cfg.max_tokens
        self._response_field = cfg.response_field
        self._on_error = cfg.on_error

        # Schema from config
        assert cfg.schema_config is not None
        schema = create_schema_from_config(
            cfg.schema_config,
            f"{self.name}Schema",
            allow_coercion=False,  # Transforms do NOT coerce
        )
        self.input_schema = schema
        self.output_schema = schema

        # Recorder reference for pooled execution (set in on_start)
        self._recorder: LandscapeRecorder | None = None

        # Create pooled executor if pool_size > 1
        if cfg.pool_config is not None:
            self._executor: PooledExecutor | None = PooledExecutor(cfg.pool_config)
        else:
            self._executor = None

        # LLM client cache for pooled execution - ensures call_index uniqueness across retries
        # Each state_id gets its own client with monotonically increasing call indices
        self._llm_clients: dict[str, AuditedLLMClient] = {}
        self._llm_clients_lock = Lock()
        # Cache underlying Azure clients to avoid recreating them
        self._underlying_client: AzureOpenAI | None = None

    def on_start(self, ctx: PluginContext) -> None:
        """Capture recorder reference for pooled execution.

        In pooled mode, _process_single_with_state() is called from worker
        threads that don't have access to PluginContext. This captures the
        recorder reference at pipeline start so it can be used later.
        """
        self._recorder = ctx.landscape

    def process(
        self,
        row: dict[str, Any] | list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process row(s) through Azure OpenAI.

        When is_batch_aware=True and used in aggregation, receives list[dict].
        Otherwise receives single dict.

        Routes to pooled or sequential execution based on pool_size config.

        Error handling follows Three-Tier Trust Model:
        1. Template rendering (THEIR DATA) - wrap, return error
        2. LLM call (EXTERNAL) - wrap, return error
        3. Internal logic (OUR CODE) - let crash

        Args:
            row: Single row dict OR list of row dicts (batch aggregation)
            ctx: Plugin context with landscape and state_id

        Returns:
            TransformResult with processed row(s) or error
        """
        # Dispatch to batch processing if given a list
        # NOTE: This isinstance check is legitimate polymorphic dispatch for
        # batch-aware transforms, not defensive programming to hide bugs.
        if isinstance(row, list):
            return self._process_batch(row, ctx)

        # Route to pooled execution if configured (single row)
        if self._executor is not None:
            if ctx.landscape is None or ctx.state_id is None:
                raise RuntimeError(
                    "Pooled execution requires landscape recorder and state_id. Ensure transform is executed through the engine."
                )
            row_ctx = RowContext(row=row, state_id=ctx.state_id, row_index=0)
            try:
                results = self._executor.execute_batch(
                    contexts=[row_ctx],
                    process_fn=self._process_single_with_state,
                )
                return results[0]
            finally:
                # Evict cached client after row completes to prevent unbounded memory growth
                # The client is only needed during retry loops within execute_batch()
                with self._llm_clients_lock:
                    self._llm_clients.pop(ctx.state_id, None)

        # Sequential execution path
        # 1. Render template with row data (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
                }
            )

        # 2. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 3. Create audited LLM client (self-contained)
        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError(
                "Azure LLM transform requires landscape recorder and state_id. Ensure transform is executed through the engine."
            )

        # Import here to avoid hard dependency on openai package
        from openai import AzureOpenAI

        underlying_client = AzureOpenAI(
            azure_endpoint=self._azure_endpoint,
            api_key=self._azure_api_key,
            api_version=self._azure_api_version,
        )

        llm_client = AuditedLLMClient(
            recorder=ctx.landscape,
            state_id=ctx.state_id,
            underlying_client=underlying_client,
            provider="azure",
        )

        # 4. Call LLM (EXTERNAL - wrap)
        try:
            response = llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            return TransformResult.error(
                {"reason": "rate_limited", "error": str(e)},
                retryable=True,
            )
        except LLMClientError as e:
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=e.retryable,
            )

        # 5. Build output row (OUR CODE - let exceptions crash)
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_usage"] = response.usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_model"] = response.model

        return TransformResult.success(output)

    def _get_underlying_client(self) -> AzureOpenAI:
        """Get or create the underlying Azure OpenAI client.

        The underlying client is stateless and can be shared across all calls.
        """
        if self._underlying_client is None:
            # Import here to avoid hard dependency on openai package
            from openai import AzureOpenAI

            self._underlying_client = AzureOpenAI(
                azure_endpoint=self._azure_endpoint,
                api_key=self._azure_api_key,
                api_version=self._azure_api_version,
            )
        return self._underlying_client

    def _get_llm_client(self, state_id: str) -> AuditedLLMClient:
        """Get or create LLM client for a state_id.

        Clients are cached to preserve call_index across retries.
        This ensures uniqueness of (state_id, call_index) even when
        the pooled executor retries after CapacityError.

        Thread-safe: multiple workers can call this concurrently.
        """
        with self._llm_clients_lock:
            if state_id not in self._llm_clients:
                if self._recorder is None:
                    raise RuntimeError("Azure transform requires recorder. Ensure on_start was called.")
                self._llm_clients[state_id] = AuditedLLMClient(
                    recorder=self._recorder,
                    state_id=state_id,
                    underlying_client=self._get_underlying_client(),
                    provider="azure",
                )
            return self._llm_clients[state_id]

    def _process_single_with_state(self, row: dict[str, Any], state_id: str) -> TransformResult:
        """Process a single row via Azure OpenAI with explicit state_id.

        This is used by the pooled executor where each row has its own state.
        Uses cached LLM clients to ensure call_index uniqueness across retries.

        Raises:
            CapacityError: On rate limit errors (for pooled retry)
        """
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
                    "template_source": self._template.template_source,
                }
            )

        # 2. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 3. Get cached LLM client (preserves call_index across retries)
        llm_client = self._get_llm_client(state_id)

        # 4. Call LLM (EXTERNAL - wrap, raise CapacityError for pooled retry)
        try:
            response = llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            # Convert to CapacityError for pooled executor retry
            raise CapacityError(429, str(e)) from e
        except LLMClientError as e:
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=False,
            )

        # 5. Build output row
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_usage"] = response.usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_model"] = response.model

        return TransformResult.success(output)

    def _process_batch(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch of rows with parallel execution via PooledExecutor.

        Called when transform is used as aggregation node and trigger fires.
        All rows share the same state_id; call_index provides audit uniqueness.

        Args:
            rows: List of row dicts from aggregation buffer
            ctx: Plugin context with shared state_id for entire batch

        Returns:
            TransformResult.success_multi() with one output row per input
        """
        if not rows:
            return TransformResult.success({"batch_empty": True, "row_count": 0})

        if ctx.landscape is None or ctx.state_id is None:
            raise RuntimeError(
                "Batch processing requires landscape recorder and state_id. Ensure transform is executed through the engine."
            )

        # Ensure we have an executor for parallel processing
        if self._executor is None:
            # Fallback: process sequentially if no pool configured
            return self._process_batch_sequential(rows, ctx)

        # Create contexts - all rows share same state_id (call_index provides uniqueness)
        contexts = [RowContext(row=row, state_id=ctx.state_id, row_index=i) for i, row in enumerate(rows)]

        # Execute all rows in parallel
        try:
            results = self._executor.execute_batch(
                contexts=contexts,
                process_fn=self._process_single_with_state,
            )
        finally:
            # Clean up cached clients
            with self._llm_clients_lock:
                self._llm_clients.pop(ctx.state_id, None)

        # Assemble output with per-row error tracking
        return self._assemble_batch_results(rows, results)

    def _process_batch_sequential(
        self,
        rows: list[dict[str, Any]],
        ctx: PluginContext,
    ) -> TransformResult:
        """Fallback for batch processing without executor (pool_size=1).

        Processes rows one at a time using existing sequential logic.
        """
        results: list[TransformResult] = []
        for row in rows:
            # Use the single-row sequential path
            result = self._process_sequential(row, ctx)
            results.append(result)
        return self._assemble_batch_results(rows, results)

    def _process_sequential(
        self,
        row: dict[str, Any],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process a single row sequentially (extracted from process()).

        This is the existing sequential logic, extracted for reuse.
        """
        # 1. Render template (THEIR DATA - wrap)
        try:
            rendered = self._template.render_with_metadata(row)
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "error": str(e),
                    "template_hash": self._template.template_hash,
                    "template_source": self._template.template_source,
                }
            )

        # 2. Build messages
        messages: list[dict[str, str]] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        # 3. Create audited LLM client (self-contained)
        # NOTE: ctx.landscape and ctx.state_id are guaranteed non-None here
        # because _process_sequential is only called from _process_batch_sequential,
        # which is only called from _process_batch after the None check.
        assert ctx.landscape is not None and ctx.state_id is not None

        # Import here to avoid hard dependency on openai package
        from openai import AzureOpenAI

        underlying_client = AzureOpenAI(
            azure_endpoint=self._azure_endpoint,
            api_key=self._azure_api_key,
            api_version=self._azure_api_version,
        )

        llm_client = AuditedLLMClient(
            recorder=ctx.landscape,
            state_id=ctx.state_id,
            underlying_client=underlying_client,
            provider="azure",
        )

        # 4. Call LLM (EXTERNAL - wrap)
        try:
            response = llm_client.chat_completion(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except RateLimitError as e:
            return TransformResult.error(
                {"reason": "rate_limited", "error": str(e)},
                retryable=True,
            )
        except LLMClientError as e:
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=e.retryable,
            )

        # 5. Build output row (OUR CODE - let exceptions crash)
        output = dict(row)
        output[self._response_field] = response.content
        output[f"{self._response_field}_usage"] = response.usage
        output[f"{self._response_field}_template_hash"] = rendered.template_hash
        output[f"{self._response_field}_variables_hash"] = rendered.variables_hash
        output[f"{self._response_field}_template_source"] = rendered.template_source
        output[f"{self._response_field}_lookup_hash"] = rendered.lookup_hash
        output[f"{self._response_field}_lookup_source"] = rendered.lookup_source
        output[f"{self._response_field}_model"] = response.model

        return TransformResult.success(output)

    def _assemble_batch_results(
        self,
        rows: list[dict[str, Any]],
        results: list[TransformResult],
    ) -> TransformResult:
        """Assemble batch results with per-row error tracking.

        Follows AzureBatchLLMTransform pattern: include all rows in output,
        mark failures with {response_field}_error instead of failing entire batch.
        """
        output_rows: list[dict[str, Any]] = []
        all_failed = True

        for i, (row, result) in enumerate(zip(rows, results, strict=True)):
            output_row = dict(row)

            if result.status == "success" and result.row is not None:
                all_failed = False
                # Copy response fields from result
                output_row[self._response_field] = result.row.get(self._response_field)
                output_row[f"{self._response_field}_usage"] = result.row.get(f"{self._response_field}_usage")
                output_row[f"{self._response_field}_template_hash"] = result.row.get(f"{self._response_field}_template_hash")
                output_row[f"{self._response_field}_variables_hash"] = result.row.get(f"{self._response_field}_variables_hash")
                output_row[f"{self._response_field}_template_source"] = result.row.get(f"{self._response_field}_template_source")
                output_row[f"{self._response_field}_lookup_hash"] = result.row.get(f"{self._response_field}_lookup_hash")
                output_row[f"{self._response_field}_lookup_source"] = result.row.get(f"{self._response_field}_lookup_source")
                output_row[f"{self._response_field}_model"] = result.row.get(f"{self._response_field}_model")
            else:
                # Per-row error tracking - don't fail entire batch
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = result.reason or {
                    "reason": "unknown_error",
                    "row_index": i,
                }

            output_rows.append(output_row)

        # Only return error if ALL rows failed
        if all_failed and output_rows:
            return TransformResult.error(
                {
                    "reason": "all_rows_failed",
                    "row_count": len(rows),
                }
            )

        return TransformResult.success_multi(output_rows)

    @property
    def azure_config(self) -> dict[str, Any]:
        """Azure configuration (for reference/debugging).

        Returns:
            Dict containing endpoint, api_version, and provider
        """
        return {
            "endpoint": self._azure_endpoint,
            "api_version": self._azure_api_version,
            "provider": "azure",
        }

    @property
    def deployment_name(self) -> str:
        """Azure deployment name (used as model in API calls)."""
        return self._deployment_name

    def close(self) -> None:
        """Release resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._recorder = None
        # Clear cached LLM clients
        with self._llm_clients_lock:
            self._llm_clients.clear()
        self._underlying_client = None
