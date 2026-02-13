# src/elspeth/plugins/llm/azure_batch.py
"""Azure OpenAI Batch API transform - 50% cost savings for high volume.

Uses two-phase checkpoint approach:
1. SUBMIT: Render templates, build JSONL, upload, submit batch, checkpoint batch_id
2. CHECK/COMPLETE: Resume with checkpoint, check status, download results or wait

Benefits:
- 50% cost reduction vs real-time API
- Crash recovery via checkpointed batch_id
- Resource efficiency (no blocking waits)

Requires Azure OpenAI Batch API (GA as of Azure API version 2024-10-21).
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import uuid
from datetime import UTC, datetime
from threading import Lock
from typing import Any

from pydantic import Field

from elspeth.contracts import BatchPendingError, CallStatus, CallType, Determinism, RowErrorEntry, TransformErrorReason, TransformResult
from elspeth.contracts.plugin_context import PluginContext
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.base import BaseTransform
from elspeth.plugins.config_base import TransformDataConfig
from elspeth.plugins.llm import get_llm_audit_fields, get_llm_guaranteed_fields
from elspeth.plugins.llm.templates import PromptTemplate, TemplateError
from elspeth.plugins.llm.tracing import (
    LangfuseTracingConfig,
    TracingConfig,
    parse_tracing_config,
    validate_tracing_config,
)
from elspeth.plugins.schema_factory import create_schema_from_config


class AzureBatchConfig(TransformDataConfig):
    """Azure Batch API-specific configuration.

    Extends TransformDataConfig with Azure-specific settings for batch processing.

    Required fields:
        deployment_name: Azure deployment name (used as model identifier)
        endpoint: Azure OpenAI endpoint URL
        api_key: Azure OpenAI API key
        template: Jinja2 prompt template

    Optional fields:
        api_version: Azure API version (default: 2024-10-21)
        system_prompt: Optional system message
        temperature: Sampling temperature (default: 0.0 for determinism)
        max_tokens: Maximum response tokens
        response_field: Output field name for LLM response (default: llm_response)
        poll_interval_seconds: Batch status check interval (default: 300s)
        max_wait_hours: Maximum batch wait time (default: 24h)
        on_error: Sink for failed rows (optional)
    """

    deployment_name: str = Field(..., description="Azure deployment name")
    endpoint: str = Field(..., description="Azure OpenAI endpoint URL")
    api_key: str = Field(..., description="Azure OpenAI API key")
    api_version: str = Field(default="2024-10-21", description="Azure API version")

    template: str = Field(..., description="Jinja2 prompt template")
    system_prompt: str | None = Field(None, description="Optional system prompt")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int | None = Field(None, gt=0, description="Maximum response tokens")
    response_field: str = Field("llm_response", description="Output field name for LLM response")

    poll_interval_seconds: int = Field(300, ge=1, description="Batch status check interval in seconds")
    max_wait_hours: int = Field(24, ge=1, le=24, description="Maximum batch wait time in hours")

    # Template metadata fields for audit trail (matching LLMConfig)
    lookup: dict[str, Any] | None = Field(None, description="Lookup data loaded from YAML file")
    template_source: str | None = Field(None, description="Template file path for audit (None if inline)")
    lookup_source: str | None = Field(None, description="Lookup file path for audit (None if no lookup)")
    system_prompt_source: str | None = Field(None, description="System prompt file path for audit (None if inline)")

    # Tier 2: Plugin-internal tracing (optional, Langfuse only)
    # Azure AI tracing is NOT supported for batch API - it auto-instruments the OpenAI SDK
    # for real-time calls, but batch jobs run asynchronously in Azure's infrastructure.
    tracing: dict[str, Any] | None = Field(
        default=None,
        description="Tier 2 tracing configuration (langfuse only - azure_ai not supported for batch API)",
    )


class AzureBatchLLMTransform(BaseTransform):
    """Batch LLM transform using Azure OpenAI Batch API.

    Uses two-phase checkpoint approach for crash recovery:

    Phase 1 (Submit):
        1. Render all templates for input rows
        2. Build JSONL request file
        3. Upload to Azure Blob storage
        4. Create batch job
        5. CHECKPOINT: Save batch_id immediately
        6. Raise BatchPendingError("submitted")

    Phase 2 (Check/Complete):
        1. Check checkpoint - if batch_id exists, resume
        2. Check Azure batch status:
           - "completed" -> Download results, return success
           - "failed" -> Return TransformResult.error()
           - "in_progress" -> Raise BatchPendingError again

    Benefits:
    - 50% cost reduction vs real-time API
    - Crash recovery via checkpointed batch_id
    - Resource efficiency (no blocking waits)

    Configuration example:
        transforms:
          - plugin: azure_batch_llm
            options:
              deployment_name: "gpt-4o-batch"
              endpoint: "${AZURE_OPENAI_ENDPOINT}"
              api_key: "${AZURE_OPENAI_KEY}"
              template: |
                Analyze: {{ row.text }}
              schema:
                mode: observed
              poll_interval_seconds: 300
    """

    name = "azure_batch_llm"
    is_batch_aware = True  # Engine passes list[dict] for batch processing

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Azure Batch LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        cfg = AzureBatchConfig.from_dict(config)
        self._deployment_name = cfg.deployment_name
        self._endpoint = cfg.endpoint.rstrip("/")
        self._api_key: str | None = cfg.api_key
        self._api_version = cfg.api_version
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
        self._poll_interval = cfg.poll_interval_seconds
        self._max_wait_hours = cfg.max_wait_hours

        # Schema from config (TransformDataConfig guarantees schema_config is not None)
        schema_config = cfg.schema_config
        schema = create_schema_from_config(
            schema_config,
            "AzureBatchSchema",
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

        # Azure OpenAI client (lazy init)
        self._client: Any = None
        self._client_lock = Lock()

        # Tier 2: Plugin-internal tracing (Langfuse only for batch API)
        self._tracing_config: TracingConfig | None = parse_tracing_config(cfg.tracing)
        self._tracing_active: bool = False
        self._langfuse_client: Any = None  # Langfuse client if configured

    def on_start(self, ctx: PluginContext) -> None:
        """Initialize tracing if configured.

        Called by the engine at pipeline start. Initializes Tier 2 tracing.
        """
        if self._tracing_config is not None:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider.

        Azure Batch API submits jobs asynchronously - we can only trace at
        the job level, not individual LLM calls. Azure AI auto-instrumentation
        is NOT supported because the OpenAI SDK isn't used for the actual
        LLM inference (that happens in Azure's batch infrastructure).
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
                # Azure AI tracing NOT supported for batch API
                logger.warning(
                    "Azure AI tracing not supported for Azure Batch API",
                    provider="azure_ai",
                    hint="Azure Batch jobs run asynchronously in Azure infrastructure - use Langfuse for job-level tracing instead",
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
        """Initialize Langfuse tracing for batch job tracking (v3 API).

        Langfuse can trace batch jobs at the job level (submit/complete),
        not per-row (since rows are processed by Azure infrastructure).
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
                "Langfuse tracing initialized for Azure Batch (v3)",
                provider="langfuse",
                host=cfg.host,
                tracing_enabled=cfg.tracing_enabled,
                note="Job-level tracing only - individual row tracing not available",
            )

        except ImportError:
            logger.warning(
                "Langfuse tracing requested but package not installed",
                provider="langfuse",
                hint="Install with: uv pip install elspeth[tracing-langfuse]",
            )

    def _record_langfuse_batch_job(
        self,
        batch_id: str,
        row_count: int,
        latency_ms: float,
        status: str,
        error: str | None = None,
    ) -> None:
        """Record a batch job in Langfuse using v3 nested context managers.

        Azure Batch API processes rows asynchronously in Azure's infrastructure,
        so we cannot trace individual LLM calls. We record the job as a single
        span with aggregate metadata.

        Args:
            batch_id: Azure batch job ID
            row_count: Number of rows in the batch
            latency_ms: Total job latency in milliseconds
            status: Job completion status ("completed", "failed", etc.)
            error: Error message if job failed
        """
        if not self._tracing_active or self._langfuse_client is None:
            return

        try:
            with self._langfuse_client.start_as_current_observation(
                as_type="span",
                name=f"elspeth.{self.name}",
                metadata={
                    "batch_id": batch_id,
                    "plugin": self.name,
                    "model": self._deployment_name,
                },
            ):
                span_metadata: dict[str, Any] = {
                    "batch_id": batch_id,
                    "row_count": row_count,
                    "latency_ms": latency_ms,
                    "status": status,
                    "note": "Individual row tracing not available for Azure Batch API",
                }
                if error:
                    span_metadata["error"] = error

                with self._langfuse_client.start_as_current_observation(
                    as_type="span",
                    name="azure_batch_job",
                ) as inner_span:
                    inner_span.update(metadata=span_metadata)

        except Exception as e:
            import structlog

            logger = structlog.get_logger(__name__)
            logger.warning("Failed to record Langfuse batch job", error=str(e))

    def _get_client(self) -> Any:
        """Lazy-initialize Azure OpenAI client.

        Thread-safe: protected by _client_lock to prevent duplicate
        client creation if future pooling adds concurrency.

        Returns:
            openai.AzureOpenAI client instance
        """
        with self._client_lock:
            if self._client is None:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    azure_endpoint=self._endpoint,
                    api_key=self._api_key,
                    api_version=self._api_version,
                )
                # Clear plaintext key — SDK client holds its own copy internally
                self._api_key = None
            return self._client

    def process(
        self,
        row: PipelineRow | list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with checkpoint-based recovery.

        When is_batch_aware=True and configured as aggregation, the engine passes list[PipelineRow].
        For single-row fallback (non-aggregation), the engine passes PipelineRow.

        Args:
            row: Single PipelineRow OR list[PipelineRow] (batch mode)
            ctx: Plugin context with checkpoint support

        Returns:
            TransformResult with processed rows or error

        Raises:
            BatchPendingError: When batch is submitted but not complete
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

        For single rows, we still use the batch API but with just one request.
        This ensures consistent behavior and audit trail.

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
            # Empty rows from empty batch - shouldn't happen for single row
            # row is already PipelineRow (the input)
            return TransformResult.success(
                row,
                success_reason={"action": "passthrough"},
            )

    def _process_batch(
        self,
        rows: list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        """Process batch with two-phase checkpoint approach.

        Phase 1: Submit fresh batch
        Phase 2: Resume with checkpoint, check status

        Args:
            rows: List of row dicts to process
            ctx: Plugin context with checkpoint support

        Returns:
            TransformResult with all processed rows

        Raises:
            BatchPendingError: When batch is pending completion
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

        # Check checkpoint for resume
        checkpoint = self._get_checkpoint(ctx)

        if checkpoint is not None:
            if "batch_id" not in checkpoint:
                raise RuntimeError(
                    f"Checkpoint missing required 'batch_id' for AzureBatchLLMTransform. Checkpoint keys: {sorted(checkpoint.keys())}"
                )
            # PHASE 2: Resume - batch already submitted
            return self._check_batch_status(checkpoint, rows, ctx)

        # PHASE 1: Fresh batch - submit new
        return self._submit_batch(rows, ctx)

    def _get_checkpoint(self, ctx: PluginContext) -> dict[str, Any] | None:
        """Get checkpoint state from context.

        Args:
            ctx: Plugin context

        Returns:
            Checkpoint dict or None if no checkpoint
        """
        return ctx.get_checkpoint()

    def _update_checkpoint(self, ctx: PluginContext, data: dict[str, Any]) -> None:
        """Update checkpoint state.

        Args:
            ctx: Plugin context
            data: Checkpoint data to save
        """
        ctx.update_checkpoint(data)

    def _clear_checkpoint(self, ctx: PluginContext) -> None:
        """Clear checkpoint state.

        Args:
            ctx: Plugin context
        """
        ctx.clear_checkpoint()

    def _submit_batch(
        self,
        rows: list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        """Submit new batch to Azure Batch API.

        Args:
            rows: Rows to process
            ctx: Plugin context

        Returns:
            Never returns - always raises BatchPendingError

        Raises:
            BatchPendingError: After successful submission
        """
        # 1. Render templates for all rows, track failures
        requests: list[dict[str, Any]] = []
        row_mapping: dict[str, dict[str, Any]] = {}  # custom_id -> {index, variables_hash}
        template_errors: list[tuple[int, str]] = []  # (index, error)

        for idx, row in enumerate(rows):
            custom_id = f"row-{idx}-{uuid.uuid4().hex[:8]}"

            try:
                rendered = self._template.render_with_metadata(row, contract=row.contract)
            except TemplateError as e:
                template_errors.append((idx, str(e)))
                continue

            # Build messages
            messages: list[dict[str, str]] = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": rendered.prompt})

            # Build batch request in Azure Batch API format
            request: dict[str, Any] = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": self._deployment_name,
                    "messages": messages,
                    "temperature": self._temperature,
                },
            }
            if self._max_tokens is not None:
                request["body"]["max_tokens"] = self._max_tokens

            requests.append(request)
            row_mapping[custom_id] = {
                "index": idx,
                "variables_hash": rendered.variables_hash,
            }

        # Build request lookup for audit recording (custom_id -> request body)
        # This allows the audit trail to record exactly what was sent to the LLM
        requests_by_id = {req["custom_id"]: req["body"] for req in requests}

        # If ALL rows failed template rendering, return error immediately
        if not requests:
            return TransformResult.error(
                {
                    "reason": "all_templates_failed",
                    "template_errors": [{"row_index": idx, "error": err} for idx, err in template_errors],
                }
            )

        # 2. Build JSONL content
        jsonl_content = "\n".join(json.dumps(req) for req in requests)

        # 3. Upload and create batch
        client = self._get_client()

        # Upload JSONL file (with audit recording)
        file_bytes = io.BytesIO(jsonl_content.encode("utf-8"))
        jsonl_bytes = jsonl_content.encode("utf-8")
        jsonl_hash = hashlib.sha256(jsonl_bytes).hexdigest()
        upload_request = {
            "operation": "files.create",
            "filename": "batch_input.jsonl",
            "purpose": "batch",
            "content_sha256": jsonl_hash,
            "content_size": len(jsonl_bytes),
            "row_count": len(requests),
        }
        start = time.perf_counter()
        try:
            batch_file = client.files.create(
                file=("batch_input.jsonl", file_bytes),
                purpose="batch",
            )
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=upload_request,
                response_data={"file_id": batch_file.id, "status": batch_file.status},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=upload_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            return TransformResult.error(
                {
                    "reason": "file_upload_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                retryable=True,  # Network/auth errors are retryable
            )

        # Create batch job (with audit recording)
        batch_request = {
            "operation": "batches.create",
            "input_file_id": batch_file.id,
            "endpoint": "/chat/completions",
            "completion_window": "24h",
        }
        start = time.perf_counter()
        try:
            batch = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/chat/completions",
                completion_window="24h",
            )
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=batch_request,
                response_data={"batch_id": batch.id, "status": batch.status},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=batch_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            return TransformResult.error(
                {
                    "reason": "batch_create_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
                retryable=True,  # Network/auth errors are retryable
            )

        # 4. CHECKPOINT immediately after submit
        checkpoint_data = {
            "batch_id": batch.id,
            "input_file_id": batch_file.id,
            "row_mapping": row_mapping,
            "template_errors": template_errors,
            "submitted_at": datetime.now(UTC).isoformat(),
            "row_count": len(rows),
            "requests": requests_by_id,  # BUG-AZURE-01: Store original requests for audit recording
        }
        self._update_checkpoint(ctx, checkpoint_data)

        # 5. Raise BatchPendingError - engine handles retry scheduling
        # Include checkpoint and node_id so caller can persist and restore
        raise BatchPendingError(
            batch.id,
            "submitted",
            check_after_seconds=self._poll_interval,
            checkpoint=checkpoint_data,
            node_id=self.node_id,
        )

    def _check_batch_status(
        self,
        checkpoint: dict[str, Any],
        rows: list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        """Check batch status and complete if ready.

        Args:
            checkpoint: Checkpoint data with batch_id
            rows: Original rows (needed for output assembly)
            ctx: Plugin context

        Returns:
            TransformResult with results if complete

        Raises:
            BatchPendingError: If batch still in progress
        """
        batch_id = checkpoint["batch_id"]
        client = self._get_client()

        # Check batch status (with audit recording)
        retrieve_request = {
            "operation": "batches.retrieve",
            "batch_id": batch_id,
        }
        start = time.perf_counter()
        try:
            batch = client.batches.retrieve(batch_id)
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=retrieve_request,
                response_data={
                    "batch_id": batch.id,
                    "status": batch.status,
                    "output_file_id": getattr(batch, "output_file_id", None),  # Tier 3: SDK attr may vary by version
                    "error_file_id": getattr(batch, "error_file_id", None),  # Tier 3: SDK attr may vary by version
                },
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=retrieve_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            # DON'T clear checkpoint - batch exists on Azure, retry should resume checking
            return TransformResult.error(
                {
                    "reason": "batch_retrieve_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "batch_id": batch_id,
                },
                retryable=True,  # Transient failure - retry status check
            )

        if batch.status == "completed":
            # Calculate latency from submission to completion
            submitted_at_str = checkpoint.get("submitted_at")
            if submitted_at_str:
                submitted_at = datetime.fromisoformat(submitted_at_str)
                latency_ms = (datetime.now(UTC) - submitted_at).total_seconds() * 1000
            else:
                latency_ms = 0.0

            # Record to Langfuse (job-level tracing)
            self._record_langfuse_batch_job(
                batch_id=batch_id,
                row_count=checkpoint.get("row_count", len(rows)),
                latency_ms=latency_ms,
                status="completed",
            )

            # Download results and assemble output
            return self._download_results(batch, checkpoint, rows, ctx)

        elif batch.status == "failed":
            # Batch failed - clear checkpoint and return error
            self._clear_checkpoint(ctx)

            error_info: TransformErrorReason = {
                "reason": "batch_failed",
                "batch_id": batch_id,
            }
            error_message = None
            if hasattr(batch, "errors") and batch.errors:  # Tier 3: SDK errors attr is optional
                error_info["errors"] = [{"message": e.message, "error_type": e.code} for e in batch.errors.data]
                error_message = "; ".join(e.message for e in batch.errors.data)

            # Calculate latency for failed batch
            submitted_at_str = checkpoint.get("submitted_at")
            if submitted_at_str:
                submitted_at = datetime.fromisoformat(submitted_at_str)
                latency_ms = (datetime.now(UTC) - submitted_at).total_seconds() * 1000
            else:
                latency_ms = 0.0

            # Record failure to Langfuse
            self._record_langfuse_batch_job(
                batch_id=batch_id,
                row_count=checkpoint.get("row_count", len(rows)),
                latency_ms=latency_ms,
                status="failed",
                error=error_message,
            )

            return TransformResult.error(error_info)

        elif batch.status == "cancelled":
            # Batch was cancelled - clear checkpoint and return error
            self._clear_checkpoint(ctx)

            # Calculate latency for cancelled batch
            submitted_at_str = checkpoint.get("submitted_at")
            if submitted_at_str:
                submitted_at = datetime.fromisoformat(submitted_at_str)
                latency_ms = (datetime.now(UTC) - submitted_at).total_seconds() * 1000
            else:
                latency_ms = 0.0

            # Record cancellation to Langfuse
            self._record_langfuse_batch_job(
                batch_id=batch_id,
                row_count=checkpoint.get("row_count", len(rows)),
                latency_ms=latency_ms,
                status="cancelled",
            )

            return TransformResult.error(
                {
                    "reason": "batch_cancelled",
                    "batch_id": batch_id,
                }
            )

        elif batch.status == "expired":
            # Batch expired (exceeded 24h) - clear checkpoint and return error
            self._clear_checkpoint(ctx)
            return TransformResult.error(
                {
                    "reason": "batch_expired",
                    "batch_id": batch_id,
                }
            )

        else:
            # Still processing (validating, in_progress, finalizing)
            # Check if we've exceeded max wait time
            submitted_at_str = checkpoint["submitted_at"]
            submitted_at = datetime.fromisoformat(submitted_at_str)
            elapsed_hours = (datetime.now(UTC) - submitted_at).total_seconds() / 3600

            if elapsed_hours > self._max_wait_hours:
                self._clear_checkpoint(ctx)
                return TransformResult.error(
                    {
                        "reason": "batch_timeout",
                        "batch_id": batch_id,
                        "elapsed_hours": elapsed_hours,
                        "max_wait_hours": self._max_wait_hours,
                    }
                )

            # Still waiting - raise BatchPendingError for retry
            # Include checkpoint and node_id so caller can persist and restore
            raise BatchPendingError(
                batch_id,
                batch.status,
                check_after_seconds=self._poll_interval,
                checkpoint=checkpoint,
                node_id=self.node_id,
            )

    def _download_results(
        self,
        batch: Any,
        checkpoint: dict[str, Any],
        rows: list[PipelineRow],
        ctx: PluginContext,
    ) -> TransformResult:
        """Download batch results and assemble output rows.

        Args:
            batch: Completed Azure batch object
            checkpoint: Checkpoint data with row mapping
            rows: Original input rows (list[PipelineRow])
            ctx: Plugin context

        Returns:
            TransformResult with all processed rows
        """
        client = self._get_client()
        if "row_mapping" not in checkpoint:
            raise RuntimeError("Checkpoint missing required 'row_mapping' for AzureBatchLLMTransform.")
        if "template_errors" not in checkpoint:
            raise RuntimeError("Checkpoint missing required 'template_errors' for AzureBatchLLMTransform.")
        row_mapping: dict[str, dict[str, Any]] = checkpoint["row_mapping"]
        template_errors: list[tuple[int, str]] = checkpoint["template_errors"]

        # Download output file (with audit recording)
        output_file_id = batch.output_file_id
        download_request = {
            "operation": "files.content",
            "file_id": output_file_id,
        }
        start = time.perf_counter()
        try:
            output_content = client.files.content(output_file_id)
            output_text = output_content.text
            output_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=download_request,
                response_data={
                    "file_id": output_file_id,
                    "content_sha256": output_hash,
                    "content_length": len(output_text),
                },
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
        except Exception as e:
            # External API failure - record error and return structured result
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=download_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=(time.perf_counter() - start) * 1000,
                provider="azure",
            )
            # DON'T clear checkpoint - batch completed on Azure, retry should re-attempt download
            return TransformResult.error(
                {
                    "reason": "file_download_failed",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "output_file_id": output_file_id,
                },
                retryable=True,  # Transient failure - retry download
            )

        # Download error file if present (partial batch failures)
        # Azure Batch API puts per-request errors in a separate error_file_id
        # Tier 3 boundary: SDK must expose this attribute — None = no errors,
        # missing attribute = SDK version mismatch (crash, don't silently skip)
        if not hasattr(batch, "error_file_id"):
            raise RuntimeError(
                f"Azure batch object missing 'error_file_id' attribute. "
                f"Ensure Azure OpenAI SDK version >= 1.14.0 supports batch error files. "
                f"batch_id={batch.id}"
            )
        error_file_id = batch.error_file_id
        if error_file_id is not None:
            error_download_request = {
                "operation": "files.content",
                "file_id": error_file_id,
                "file_type": "error",
            }
            start = time.perf_counter()
            try:
                error_content = client.files.content(error_file_id)
                error_text = error_content.text
                error_hash = hashlib.sha256(error_text.encode("utf-8")).hexdigest()
                ctx.record_call(
                    call_type=CallType.HTTP,
                    status=CallStatus.SUCCESS,
                    request_data=error_download_request,
                    response_data={
                        "file_id": error_file_id,
                        "content_sha256": error_hash,
                        "content_length": len(error_text),
                    },
                    latency_ms=(time.perf_counter() - start) * 1000,
                    provider="azure",
                )
            except Exception as e:
                # Error file download failed - log but don't fail the batch
                # The output file was already downloaded successfully
                ctx.record_call(
                    call_type=CallType.HTTP,
                    status=CallStatus.ERROR,
                    request_data=error_download_request,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    latency_ms=(time.perf_counter() - start) * 1000,
                    provider="azure",
                )
                error_text = None
        else:
            error_text = None

        # Parse JSONL results (EXTERNAL DATA - wrap parsing)
        results_by_id: dict[str, dict[str, Any]] = {}
        malformed_lines: list[str] = []

        for line_num, line in enumerate(output_text.strip().split("\n"), start=1):
            if not line:
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError as e:
                # Log malformed line but continue processing other lines
                malformed_lines.append(f"Line {line_num}: JSON parse error - {e}")
                continue

            # Validate custom_id presence
            custom_id = result.get("custom_id")
            if custom_id is None:
                malformed_lines.append(f"Line {line_num}: Missing 'custom_id' field")
                continue

            # Validate custom_id is one we sent (membership check at Tier 3 boundary)
            # Azure returning an unknown custom_id would indicate a serious API issue
            if custom_id not in row_mapping:
                malformed_lines.append(f"Line {line_num}: Unknown 'custom_id': {custom_id}")
                continue

            # Tier 3 -> Tier 2 boundary: Validate structure immediately
            # Azure Batch API returns either "error" OR "response", never both
            has_error = "error" in result
            has_response = "response" in result

            if has_error and has_response:
                malformed_lines.append(f"Line {line_num}: Has both 'error' and 'response'")
                continue
            if not has_error and not has_response:
                malformed_lines.append(f"Line {line_num}: Missing both 'error' and 'response'")
                continue

            # Validate response structure if present
            if has_response:
                response = result["response"]
                if not isinstance(response, dict):
                    malformed_lines.append(f"Line {line_num}: 'response' is not a dict")
                    continue
                if "body" not in response:
                    malformed_lines.append(f"Line {line_num}: Missing 'response.body'")
                    continue

            # Now validated - store as Tier 2 data
            results_by_id[custom_id] = result

        # Parse error file if downloaded (EXTERNAL DATA - same validation as output)
        if error_text is not None:
            for line_num, line in enumerate(error_text.strip().split("\n"), start=1):
                if not line:
                    continue

                try:
                    error_result = json.loads(line)
                except json.JSONDecodeError:
                    malformed_lines.append(f"Error file line {line_num}: JSON parse error")
                    continue

                custom_id = error_result.get("custom_id")
                if custom_id is None:
                    malformed_lines.append(f"Error file line {line_num}: Missing 'custom_id'")
                    continue

                if custom_id not in row_mapping:
                    malformed_lines.append(f"Error file line {line_num}: Unknown 'custom_id': {custom_id}")
                    continue

                # Don't overwrite successful results from output file
                if custom_id not in results_by_id:
                    # Extract error body with explicit structure checking.
                    # Azure batch errors appear in two formats:
                    #   {"error": {...}}  or  {"response": {"body": {...}}}
                    # Preserve raw structure when neither matches, rather than
                    # silently falling back to an empty dict.
                    if "error" in error_result:
                        error_body = error_result["error"]
                    elif "response" in error_result and isinstance(error_result["response"], dict) and "body" in error_result["response"]:
                        error_body = error_result["response"]["body"]
                    else:
                        error_body = {
                            "reason": "unrecognized_error_format",
                            "raw_keys": list(error_result.keys()),
                            "raw_preview": str(error_result)[:500],
                        }
                    results_by_id[custom_id] = {
                        "custom_id": custom_id,
                        "error": error_body,
                    }

        # If ALL lines are malformed, fail the entire batch
        if not results_by_id and malformed_lines:
            self._clear_checkpoint(ctx)
            return TransformResult.error(
                {
                    "reason": "all_output_lines_malformed",
                    "malformed_count": len(malformed_lines),
                    "errors": list(malformed_lines[:10]),  # First 10 errors for diagnosis
                }
            )

        # Assemble output rows in original order
        output_rows: list[dict[str, Any]] = []
        row_errors: list[RowErrorEntry] = []

        # Track which rows had template errors (excluded from batch)
        template_error_indices = {idx for idx, _ in template_errors}

        # Build reverse mapping once (O(n) instead of O(n^2) lookup per row)
        idx_to_custom_id: dict[int, str] = {info["index"]: cid for cid, info in row_mapping.items()}

        for idx, row in enumerate(rows):
            if idx in template_error_indices:
                # Row had template error - include original row with error field
                error_msg = next((err for i, err in template_errors if i == idx), "Unknown error")
                output_row = row.to_dict()
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "template_rendering_failed",
                    "error": error_msg,
                }
                output_rows.append(output_row)
                continue

            # Find result by custom_id using pre-built reverse mapping
            if idx not in idx_to_custom_id:
                raise RuntimeError(f"Checkpoint row_mapping missing entry for row index {idx} in AzureBatchLLMTransform.")
            custom_id = idx_to_custom_id[idx]

            if custom_id not in results_by_id:
                # Result not found in Azure batch output - request was sent but no response received
                # This is rare but can happen with Azure Batch API edge cases
                output_row = row.to_dict()
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "result_not_found",
                    "custom_id": custom_id,
                }
                output_rows.append(output_row)
                row_errors.append({"row_index": idx, "reason": "result_not_found"})

                # Record Call for audit trail completeness - request WAS made but no response
                # Without this, explain(token_id) would show incomplete lineage
                # Access directly from checkpoint (Tier 1 data - we wrote it)
                original_request = checkpoint["requests"][custom_id]
                ctx.record_call(
                    call_type=CallType.LLM,
                    status=CallStatus.ERROR,
                    request_data={
                        "custom_id": custom_id,
                        "row_index": idx,
                        **original_request,
                    },
                    response_data=None,
                    error={"reason": "result_not_found", "custom_id": custom_id},
                    provider="azure",
                )
                continue

            result = results_by_id[custom_id]

            if "error" in result:
                # API error for this row
                output_row = row.to_dict()
                output_row[self._response_field] = None
                output_row[f"{self._response_field}_error"] = {
                    "reason": "api_error",
                    "error": result["error"],
                }
                output_rows.append(output_row)
                row_errors.append({"row_index": idx, "reason": "api_error", "error": result["error"]})
            else:
                # Success - extract response
                response = result["response"]
                body = response["body"]

                # Tier 3 boundary validation: Azure API response structure
                # Validate expected fields exist with correct types before accessing
                choices = body.get("choices")
                if not isinstance(choices, list) or len(choices) == 0:
                    # Missing or empty choices - record as validation error
                    output_row = row.to_dict()
                    output_row[self._response_field] = None
                    output_row[f"{self._response_field}_error"] = {
                        "reason": "invalid_response_structure",
                        "error": "Azure API response missing 'choices' array",
                    }
                    output_rows.append(output_row)
                    row_errors.append({"row_index": idx, "reason": "no_choices_in_response"})
                    continue

                first_choice = choices[0]
                if not isinstance(first_choice, dict):
                    output_row = row.to_dict()
                    output_row[self._response_field] = None
                    output_row[f"{self._response_field}_error"] = {
                        "reason": "invalid_response_structure",
                        "error": f"choices[0] is not a dict, got {type(first_choice).__name__}",
                    }
                    output_rows.append(output_row)
                    row_errors.append({"row_index": idx, "reason": "invalid_choice_structure"})
                    continue

                message = first_choice.get("message")
                if not isinstance(message, dict):
                    output_row = row.to_dict()
                    output_row[self._response_field] = None
                    output_row[f"{self._response_field}_error"] = {
                        "reason": "invalid_response_structure",
                        "error": f"choices[0].message is not a dict, got {type(message).__name__}",
                    }
                    output_rows.append(output_row)
                    row_errors.append({"row_index": idx, "reason": "invalid_message_structure"})
                    continue

                # Boundary validation passed - now we trust these fields
                content = message.get("content", "")  # content can be empty string, that's valid
                usage = body.get("usage", {})  # usage is optional in Azure API

                output_row = row.to_dict()
                output_row[self._response_field] = content

                # Retrieve variables_hash from checkpoint
                row_info = row_mapping[custom_id]
                variables_hash = row_info["variables_hash"]

                # Guaranteed fields (contract-stable)
                output_row[f"{self._response_field}_usage"] = usage
                output_row[f"{self._response_field}_model"] = body.get("model", self._deployment_name)

                # Audit fields (provenance metadata)
                output_row[f"{self._response_field}_template_hash"] = self._template.template_hash
                output_row[f"{self._response_field}_variables_hash"] = variables_hash
                output_row[f"{self._response_field}_template_source"] = self._template.template_source
                output_row[f"{self._response_field}_lookup_hash"] = self._template.lookup_hash
                output_row[f"{self._response_field}_lookup_source"] = self._template.lookup_source
                output_row[f"{self._response_field}_system_prompt_source"] = self._system_prompt_source

                output_rows.append(output_row)

        # Record per-row LLM calls against the batch's state
        # Uses existing state_id from context (set by AggregationExecutor)
        # Note: checkpoint["requests"] is Tier 1 data (we wrote it) - crash if missing
        requests_data = checkpoint["requests"]

        for custom_id, result in results_by_id.items():
            original_request = requests_data[custom_id]
            row_index = row_mapping[custom_id]["index"]

            # Determine call status from result (Tier 2 - validated at boundary)
            if "error" in result:
                call_status = CallStatus.ERROR
                response_data = None
                error_data = {"error": result["error"]}
            else:
                call_status = CallStatus.SUCCESS
                # response.body guaranteed by boundary validation
                response_data = result["response"]["body"]
                error_data = None

            # Record LLM call with custom_id for token mapping
            ctx.record_call(
                call_type=CallType.LLM,
                status=call_status,
                request_data={
                    "custom_id": custom_id,
                    "row_index": row_index,
                    **original_request,
                },
                response_data=response_data,
                error=error_data,
                provider="azure",
            )

        # Clear checkpoint after successful completion
        self._clear_checkpoint(ctx)

        # Return results
        if not output_rows:
            # All rows failed - return error
            return TransformResult.error(
                {
                    "reason": "all_rows_failed",
                    "row_errors": row_errors,
                }
            )

        # Create OBSERVED contract from union of ALL output row keys (not just first)
        # Error rows may have extra fields (e.g. _error) that the first row lacks
        # Infer python_type from first non-None value seen per key across all rows
        from elspeth.contracts.schema_contract import FieldContract, SchemaContract

        _PRIMITIVE_TYPES = (int, str, float, bool)
        all_keys: dict[str, type] = {}
        for r in output_rows:
            for key, value in r.items():
                if key not in all_keys:
                    all_keys[key] = type(value) if value is not None and type(value) in _PRIMITIVE_TYPES else object
                elif all_keys[key] is object and value is not None and type(value) in _PRIMITIVE_TYPES:
                    all_keys[key] = type(value)

        fields = tuple(
            FieldContract(
                normalized_name=key,
                original_name=key,
                python_type=inferred_type,
                required=False,
                source="inferred",
            )
            for key, inferred_type in all_keys.items()
        )
        output_contract = SchemaContract(mode="OBSERVED", fields=fields, locked=True)
        return TransformResult.success_multi(
            [PipelineRow(r, output_contract) for r in output_rows],
            success_reason={"action": "enriched", "fields_added": [self._response_field]},
        )

    @property
    def azure_config(self) -> dict[str, Any]:
        """Azure configuration for executor (if needed).

        Returns:
            Dict containing endpoint, api_version, and provider.
            API key is intentionally excluded to prevent accidental exposure
            in checkpoints, logs, or audit records.
        """
        return {
            "endpoint": self._endpoint,
            "api_version": self._api_version,
            "provider": "azure_batch",
        }

    @property
    def deployment_name(self) -> str:
        """Azure deployment name."""
        return self._deployment_name

    def close(self) -> None:
        """Release resources and flush tracing."""
        # Flush Tier 2 tracing if active
        if self._tracing_active:
            self._flush_tracing()

        self._client = None
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
