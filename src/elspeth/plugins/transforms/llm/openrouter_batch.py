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

import json
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, ClassVar, Literal

import httpx
import structlog
from pydantic import Field, field_validator

from elspeth.contracts import CallStatus, CallType, Determinism, TransformResult, TransformSuccessReason
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.token_usage import TokenUsage
from elspeth.contracts.value_source import (
    CatalogValueSource,
    ValueSource,
    register_value_source_plugin,
)
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.clients.http import AuditedHTTPClient
from elspeth.plugins.infrastructure.pooling import is_capacity_error
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.infrastructure.templates import TemplateError
from elspeth.plugins.transforms.llm import (
    _build_augmented_output_schema,
    build_llm_audit_metadata,
    get_llm_guaranteed_fields,
    populate_llm_operational_fields,
)
from elspeth.plugins.transforms.llm.base import LLMConfig
from elspeth.plugins.transforms.llm.langfuse import create_langfuse_tracer
from elspeth.plugins.transforms.llm.model_catalog import MODEL_CATALOG_OPENROUTER
from elspeth.plugins.transforms.llm.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OPENROUTER_BASE_URL_APPLIES_WHEN,
    normalize_openrouter_base_url,
)
from elspeth.plugins.transforms.llm.templates import PromptTemplate
from elspeth.plugins.transforms.llm.tracing import (
    TracingConfig,
    parse_tracing_config,
    validate_tracing_config,
)
from elspeth.plugins.transforms.llm.validation import reject_nonfinite_constant

_logger = structlog.get_logger(__name__)


def _warn_telemetry_before_start(event: Any) -> None:
    """Default telemetry callback before on_start() — warns instead of silently dropping."""
    _logger.warning(
        "telemetry_emit called before on_start() — event dropped",
        event_type=type(event).__name__,
    )


@dataclass(frozen=True, slots=True)
class _RowSuccess:
    """Successful single-row result in OpenRouter batch."""

    # ``Mapping`` (not ``dict``): ``freeze_fields`` replaces the bound
    # value with ``MappingProxyType``; the annotation must describe the
    # read-only runtime type callers observe. See
    # ``PooledExecutor.RowContext`` for the same pattern.
    row: Mapping[str, Any]
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        freeze_fields(self, "row")


@dataclass(frozen=True, slots=True)
class _RowFailure:
    """Failed single-row result in OpenRouter batch."""

    error: Mapping[str, Any]

    def __post_init__(self) -> None:
        freeze_fields(self, "error")


_RowOutcome = _RowSuccess | _RowFailure


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

    # OpenRouter batch configs always have provider="openrouter" — narrowed Literal prevents misconfiguration
    provider: Literal["openrouter"] = Field(default="openrouter", description="LLM provider")

    # Override base model to make it required — OpenRouter has no deployment_name fallback
    model: str = Field(..., description="Model identifier (e.g., 'openai/gpt-4o-mini')")

    api_key: str = Field(..., description="OpenRouter API key")
    base_url: str = Field(
        default=OPENROUTER_BASE_URL,
        description="OpenRouter API base URL",
    )
    timeout_seconds: float = Field(default=60.0, gt=0, description="Request timeout")

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        return normalize_openrouter_base_url(value)

    # Tier 2: Plugin-internal tracing (optional, Langfuse only)
    # Azure AI tracing is NOT supported - it auto-instruments the OpenAI SDK,
    # but OpenRouter uses HTTP directly via httpx.
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (langfuse only - azure_ai not supported)",
    )

    # Value-source declaration: ``model`` must appear in the OpenRouter
    # slice of ``litellm.model_list`` whenever ``base_url`` targets the
    # canonical OpenRouter endpoint. The ``applies_when`` predicate keeps
    # the catalog check in lock-step with the actual HTTP boundary —
    # operators redirecting to a chaos test server (errorworks /
    # chaosllm) or a private OpenAI-compatible gateway skip the catalog
    # check, since the model identifier semantics are owned by that
    # endpoint, not by litellm's OpenRouter slug list. Mirrors the
    # non-batch :attr:`OpenRouterConfig.VALUE_SOURCES` exactly so batch
    # and non-batch enforcement do not drift.
    VALUE_SOURCES: ClassVar[tuple[ValueSource, ...]] = (
        CatalogValueSource(
            field_name="model",
            catalog_id=MODEL_CATALOG_OPENROUTER,
            applies_when=OPENROUTER_BASE_URL_APPLIES_WHEN,
        ),
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
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:0139902ccd1a63ad"
    is_batch_aware = True  # Engine passes list[dict] for batch processing
    config_model = OpenRouterBatchConfig

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize OpenRouter batch LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        # Parse OpenRouter-specific config
        cfg = OpenRouterBatchConfig.from_dict(config, plugin_name=self.name)
        # Retain the typed config so the value-source compliance walker
        # can inspect ``VALUE_SOURCES`` declarations and post-validation
        # field values without scraping ``self._field_name`` destructured
        # copies. Exposed publicly via ``provider_config``.
        self._config = cfg
        self._initialize_declared_input_fields(cfg)

        # Declare output fields for centralized collision detection.
        self.declared_output_fields = frozenset(get_llm_guaranteed_fields(cfg.response_field))

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
        self.output_schema = _build_augmented_output_schema(
            base_schema_config=schema_config,
            response_field=cfg.response_field,
            schema_name=f"{self.name}OutputSchema",
        )

        # Build output schema config with field categorization
        guaranteed = get_llm_guaranteed_fields(self._response_field)

        # Merge with any existing fields from base schema
        base_guaranteed = set(schema_config.guaranteed_fields or ())
        output_fields = base_guaranteed | set(guaranteed)
        # Preserve None-vs-empty-tuple semantics: None = abstain, () = explicitly empty.
        upstream_declared = schema_config.guaranteed_fields is not None
        if upstream_declared or output_fields:
            guaranteed_fields_result = tuple(sorted(output_fields))
        else:
            guaranteed_fields_result = None

        self._output_schema_config = SchemaConfig(
            mode=schema_config.mode,
            fields=schema_config.fields,
            guaranteed_fields=guaranteed_fields_result,
            required_fields=schema_config.required_fields,
        )

        # Recorder and telemetry references (set in on_start)
        self._recorder: PluginAuditWriter | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = _warn_telemetry_before_start
        self._limiter: Any = None  # RateLimiter | NoOpLimiter | None

        # HTTP client cache — one per state_id for call_index uniqueness.
        # Each state_id gets its own AuditedHTTPClient with monotonically
        # increasing call indices, ensuring UNIQUE(state_id, call_index).
        self._http_clients: dict[str, AuditedHTTPClient] = {}
        self._http_clients_lock = Lock()

        # Tier 2: Plugin-internal tracing (Langfuse only)
        self._tracing_config: TracingConfig | None = parse_tracing_config(cfg.tracing)
        self._tracer = create_langfuse_tracer(
            transform_name=self.name,
            tracing_config=self._tracing_config,
        )

    @property
    def provider_config(self) -> OpenRouterBatchConfig:
        """Read-only accessor for the typed batch config.

        Mirrors :attr:`elspeth.plugins.transforms.llm.transform.LLMTransform.provider_config`
        so the value-source compliance walker can inspect the
        ``VALUE_SOURCES`` declaration and post-validation field values
        without reaching into ``self._config`` private state. Named
        ``provider_config`` (not ``config``) to avoid colliding with
        :attr:`BaseTransform.config` (the raw ``dict[str, Any]``).
        """
        return self._config

    def on_start(self, ctx: LifecycleContext) -> None:
        """Capture recorder, telemetry, rate limit context, and initialize tracing.

        Called by the engine at pipeline start. Captures the landscape
        recorder, run_id, telemetry callback, and rate limiter for use in
        worker threads. Also initializes Tier 2 tracing if configured.
        """
        super().on_start(ctx)
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
        if self._tracing_config is None:
            return

        # Validate configuration completeness
        errors = validate_tracing_config(self._tracing_config)
        if errors:
            raise ValueError(f"Tracing configuration errors: {'; '.join(errors)}")

        match self._tracing_config.provider:
            case "azure_ai":
                # Azure AI tracing NOT supported for OpenRouter
                raise ValueError(
                    "Azure AI tracing is not supported for OpenRouter. "
                    "Azure AI auto-instruments the OpenAI SDK; OpenRouter uses HTTP directly — "
                    "use Langfuse instead (provider: langfuse)."
                )
            case "langfuse":
                pass  # Handled by create_langfuse_tracer() in __init__
            case "none":
                pass  # No tracing
            case _:
                raise ValueError(
                    f"Unknown tracing provider '{self._tracing_config.provider}' after validation. Supported: azure_ai, langfuse, none."
                )

    def process(
        self,
        row: PipelineRow | list[PipelineRow],
        ctx: TransformContext,
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
        ctx: TransformContext,
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
        ctx: TransformContext,
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

        # Snapshot state_id for per-batch client eviction (same pattern as azure.py).
        # Aggregation flushes generate new state_ids each time, so without eviction
        # the client cache grows unboundedly.
        state_id = ctx.state_id

        # Process rows in parallel using AuditedHTTPClient (one per state_id).
        # Each worker thread gets a cached client via _get_http_client().
        results: dict[int, _RowOutcome | Exception] = {}

        try:
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
        finally:
            # Evict and close the state-scoped HTTP client after each batch completes
            # (success or failure). Without this, each flush creates a new state_id and
            # the cache grows unboundedly for long-running aggregation pipelines.
            if state_id is not None:
                with self._http_clients_lock:
                    client = self._http_clients.pop(state_id, None)
                if client is not None:
                    client.close()

        # Assemble output rows in original order
        # Every row gets an output (success or with error markers) - no rows are dropped
        output_rows: list[dict[str, Any]] = []
        finish_reason_counts: Counter[str] = Counter()
        quarantined_indices: list[int] = []
        row_errors: list[dict[str, Any]] = []

        for idx in range(len(rows)):
            if idx not in results:
                raise RuntimeError(
                    f"OpenRouter batch results missing for row index {idx}. This indicates an internal concurrency or collection bug."
                )
            result = results[idx]

            if isinstance(result, Exception):
                # StreamError escaped from _process_single_row — record to audit
                # trail so the failure is attributable (h27 fix).
                # NOTE: Not wrapped in AuditIntegrityError — per-row recording in batch
                # loop. Crashing here would lose all progress for remaining rows.
                error_payload = {
                    "reason": "transport_exception",
                    "error": str(result),
                    "error_type": type(result).__name__,
                }
                ctx.record_call(
                    call_type=CallType.LLM,
                    status=CallStatus.ERROR,
                    request_data={"row_index": idx},
                    response_data=None,
                    error=error_payload,
                    latency_ms=None,
                    provider="openrouter",
                )
                output_row = rows[idx].to_dict()
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = error_payload
                output_rows.append(output_row)
                quarantined_indices.append(idx)
                row_errors.append({"row_index": idx, **error_payload})

            elif isinstance(result, _RowFailure):
                # Row-level error from _process_single_row
                error_payload = dict(result.error)
                output_row = rows[idx].to_dict()
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = error_payload
                output_rows.append(output_row)
                quarantined_indices.append(idx)
                row_errors.append({"row_index": idx, **error_payload})

            elif isinstance(result, _RowSuccess):
                # result.row is a MappingProxyType (deep-frozen in _RowSuccess.__post_init__).
                # PipelineRow below requires exact dict type (Tier 1 guard against
                # silently coercing non-dict audit/replay data), so materialize a fresh
                # dict here — PipelineRow re-deep-freezes internally, so immutability
                # is preserved through the handoff.
                output_rows.append(dict(result.row))
                fr_key = result.finish_reason or "absent"
                finish_reason_counts[fr_key] += 1

            else:
                raise RuntimeError(f"Unexpected result type: {type(result).__name__}")

        # Create OBSERVED contract from union of ALL output row keys (not just first)
        # Error rows may have extra fields (e.g. _error) that the first row lacks
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
        output_contract = self._align_output_contract(output_contract)

        # Batch-level template provenance — shared across all rows in the batch.
        # Per-row request/response hashes are in the call audit trail.
        batch_audit = build_llm_audit_metadata(
            self._response_field,
            template_hash=self._template.template_hash,
            variables_hash=None,  # Batch-level: per-row hashes are recorded in call audit metadata
            template_source=self._template.template_source,
            lookup_hash=self._template.lookup_hash,
            lookup_source=self._template.lookup_source,
            system_prompt_source=self._system_prompt_source,
        )

        success_reason: TransformSuccessReason = {
            "action": "enriched",
            "fields_added": [self._response_field],
            "metadata": {
                "batch_size": len(output_rows),
                "finish_reason_summary": finish_reason_counts,
                **batch_audit,
            },
        }
        if quarantined_indices:
            success_reason["metadata"]["quarantined_indices"] = quarantined_indices
            success_reason["metadata"]["quarantined_count"] = len(quarantined_indices)
            success_reason["metadata"]["row_errors"] = row_errors

        return TransformResult.success_multi(
            [PipelineRow(r, output_contract) for r in output_rows],
            success_reason=success_reason,
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
                    raise RuntimeError("_recorder not initialized — _get_http_client called before begin_run()")
                self._http_clients[state_id] = AuditedHTTPClient(
                    execution=self._recorder,
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
        ctx: TransformContext,
    ) -> _RowOutcome:
        """Process a single row through OpenRouter API.

        Called by worker threads. Returns a _RowSuccess with
        the processed row dict on success, or ok=False with error details
        on failure.

        Uses AuditedHTTPClient which automatically records HTTP calls to the
        Landscape audit trail and emits telemetry events. Manual ctx.record_call()
        is only used for pre-HTTP errors (template rendering).

        Args:
            idx: Row index in batch
            row: Row to process
            ctx: Plugin context

        Returns:
            _RowSuccess on success, _RowFailure on failure
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
            return _RowFailure(error={"reason": "template_rendering_failed", "error": str(e)})

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
                audit_request_metadata={"variables_hash": rendered.variables_hash},
                token_id=row_token_id,
            )
            response.raise_for_status()

        except httpx.HTTPStatusError as e:
            # HTTP error already recorded by AuditedHTTPClient — just trace and return
            self._tracer.record_error(
                token_id=row_token_id or f"batch-idx-{idx}",
                query_name=self.name,
                prompt=rendered.prompt,
                error_message=f"HTTP {e.response.status_code}: {e}",
                model=self._model,
                extra_metadata={"row_index": idx},
            )
            return _RowFailure(
                error={
                    "reason": "api_call_failed",
                    "error": str(e),
                    "status_code": e.response.status_code,
                    "retryable": is_capacity_error(e.response.status_code),
                },
            )
        except httpx.RequestError as e:
            # Network error already recorded by AuditedHTTPClient — just trace and return
            self._tracer.record_error(
                token_id=row_token_id or f"batch-idx-{idx}",
                query_name=self.name,
                prompt=rendered.prompt,
                error_message=f"Request error: {e}",
                model=self._model,
                extra_metadata={"row_index": idx},
            )
            return _RowFailure(
                error={
                    "reason": "api_call_failed",
                    "error": str(e),
                    "retryable": False,
                },
            )

        # 4. Parse JSON response (EXTERNAL DATA - wrap, reject NaN/Infinity)
        try:
            data = json.loads(response.content, parse_constant=reject_nonfinite_constant)
        except (ValueError, TypeError) as e:
            return _RowFailure(
                error={
                    "reason": "invalid_json_response",
                    "error": str(e),
                    **({"body_preview": response.text[:500]} if response.text else {}),
                },
            )

        # 5. Validate response structure (EXTERNAL DATA - validate at boundary)
        if not isinstance(data, dict):
            return _RowFailure(
                error={
                    "reason": "invalid_json_type",
                    "expected": "object",
                    "actual": type(data).__name__,
                },
            )

        # 6. Extract content (EXTERNAL DATA - wrap)
        try:
            choices = data["choices"]
            if not choices:
                return _RowFailure(error={"reason": "empty_choices", "response": data})

            content = choices[0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            return _RowFailure(
                error={
                    "reason": "malformed_response",
                    "error": f"{type(e).__name__}: {e}",
                    "response_keys": list(data.keys()) if isinstance(data, dict) else None,
                },
            )

        raw_finish_reason = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None

        # Null content = content filtered by provider (Tier 3 boundary).
        # Tool-call responses from OpenRouter use content=null, so inspect
        # finish_reason before collapsing null content into content filtering.
        if content is None:
            if raw_finish_reason == "tool_calls":
                return _RowFailure(
                    error={
                        "reason": "unsupported_finish_reason",
                        "finish_reason": "tool_calls",
                        "error": "LLM returned tool_calls response (not supported by ELSPETH)",
                    },
                )
            return _RowFailure(
                error={
                    "reason": "content_filtered",
                    "error": "LLM returned null content (likely content-filtered by provider)",
                },
            )

        # Tier 3 boundary: validate content is string
        if not isinstance(content, str):
            return _RowFailure(
                error={
                    "reason": "malformed_content_type",
                    "error": f"Expected string content, got {type(content).__name__}",
                },
            )

        # Tier 3 boundary: validate finish_reason
        if raw_finish_reason == "tool_calls":
            return _RowFailure(
                error={
                    "reason": "unsupported_finish_reason",
                    "finish_reason": "tool_calls",
                    "error": "LLM returned tool_calls response (not supported by ELSPETH)",
                },
            )

        # Empty/whitespace content — provider returned a string but no meaningful text
        if not content.strip():
            return _RowFailure(
                error={
                    "reason": "empty_content",
                    "finish_reason": raw_finish_reason,
                    "error": f"LLM returned empty content (finish_reason={raw_finish_reason})",
                },
            )

        # Non-stop finish reasons with content indicate truncation or content filtering
        # that the provider didn't flag via null content. Record as failure.
        if raw_finish_reason is not None and raw_finish_reason not in ("stop",):
            return _RowFailure(
                error={
                    "reason": "non_stop_finish_reason",
                    "finish_reason": raw_finish_reason,
                    "error": f"LLM response has finish_reason={raw_finish_reason!r} (expected 'stop')",
                },
            )

        # Note: "usage" and "model" are optional in OpenAI/OpenRouter API responses
        # (e.g., streaming responses may omit usage). Tier 3 boundary: coerce to TokenUsage.
        usage = TokenUsage.from_dict(data.get("usage"))
        response_model = data.get("model")

        # Record to Langfuse (per-call tracing — unlike Azure Batch, we control each call)
        self._tracer.record_success(
            token_id=row_token_id or f"batch-idx-{idx}",
            query_name=self.name,
            prompt=rendered.prompt,
            response_content=content,
            model=response_model,
            usage=usage,
            extra_metadata={"row_index": idx, "model": response_model},
        )

        # 7. Build output row (OUR CODE - let exceptions crash)
        output = row.to_dict()
        output[self._response_field] = content
        populate_llm_operational_fields(
            output,
            self._response_field,
            usage=usage,
            model=response_model,
        )

        return _RowSuccess(row=output, finish_reason=raw_finish_reason)

    def close(self) -> None:
        """Release resources and flush tracing."""
        # Flush Tier 2 tracing (no-op if tracer is inactive)
        self._tracer.flush()

        # Close and clear cached HTTP clients
        with self._http_clients_lock:
            for client in self._http_clients.values():
                client.close()
            self._http_clients.clear()

        self._recorder = None


# Register opt-in for value-source compliance: the typed Pydantic config
# (OpenRouterBatchConfig) is exposed via the public ``provider_config``
# property. Mirrors the non-batch registration at transform.py:1480 so
# the walker (engine/orchestrator/preflight.py) treats batch and
# non-batch OpenRouter transforms identically.
register_value_source_plugin(OpenRouterBatchLLMTransform, config_attr="provider_config")
