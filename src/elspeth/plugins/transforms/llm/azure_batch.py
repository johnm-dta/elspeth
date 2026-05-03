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
from collections import Counter
from datetime import UTC, datetime
from threading import Lock
from typing import Any, ClassVar

import structlog
from pydantic import Field, model_validator

from elspeth.contracts import (
    BatchPendingError,
    CallStatus,
    CallType,
    Determinism,
    RowErrorEntry,
    TransformErrorReason,
    TransformResult,
    TransformSuccessReason,
)
from elspeth.contracts.batch_checkpoint import BatchCheckpointState, RowMappingEntry
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.errors import AuditIntegrityError
from elspeth.contracts.schema import SchemaConfig
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.token_usage import TokenUsage
from elspeth.contracts.value_source import (
    DerivedFromSiblingValueSource,
    ValueSource,
    register_value_source_plugin,
)
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.config_base import TransformDataConfig
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.infrastructure.templates import TemplateError
from elspeth.plugins.transforms.llm import (
    _build_augmented_output_schema,
    build_llm_audit_metadata,
    get_llm_guaranteed_fields,
    populate_llm_operational_fields,
)
from elspeth.plugins.transforms.llm.langfuse import ActiveLangfuseTracer, create_langfuse_tracer
from elspeth.plugins.transforms.llm.templates import PromptTemplate
from elspeth.plugins.transforms.llm.tracing import (
    TracingConfig,
    parse_tracing_config,
    validate_tracing_config,
)

logger = structlog.get_logger(__name__)


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

    # Mirrors ``AzureOpenAIConfig.model`` for parity with the non-batch
    # path: an operator may set ``model:`` in YAML for documentation /
    # readability, but the canonical model identifier on Azure is the
    # deployment_name. The ``_set_model_from_deployment`` validator below
    # fills the field when omitted; the value-source compliance walker
    # then enforces ``model == deployment_name`` post-validation so a
    # divergent override (a hallucinated model identifier) is rejected
    # at preflight rather than at the first per-row HTTP call.
    model: str = Field(default="", description="Model identifier (defaults to deployment_name)")

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

    @model_validator(mode="before")
    @classmethod
    def _set_model_from_deployment(cls, data: Any) -> Any:
        """Fill ``model`` from ``deployment_name`` when the operator
        leaves the field empty. Mirrors :meth:`AzureOpenAIConfig._set_model_from_deployment`
        so batch and non-batch Azure transforms share the same operator
        ergonomics (omit ``model`` and let it inherit).
        """
        if isinstance(data, dict) and not data.get("model"):
            deployment = data.get("deployment_name")
            if deployment:
                data["model"] = deployment
        return data

    # Value-source declaration: ``model`` is derived from ``deployment_name``
    # — same contract as ``AzureOpenAIConfig`` (the non-batch sibling).
    # The validator above fills the field when empty; the walker confirms
    # post-validation that ``model == deployment_name`` (or that the
    # original config left it empty, which the validator legalizes by
    # substitution). This rejects a hallucinated batch ``model`` value at
    # preflight rather than at first batch-submission HTTP call.
    VALUE_SOURCES: ClassVar[tuple[ValueSource, ...]] = (
        DerivedFromSiblingValueSource(
            field_name="model",
            sibling_field="deployment_name",
            allow_empty_default=True,
        ),
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
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:3a118aa8cdf61caa"
    is_batch_aware = True  # Engine passes list[dict] for batch processing
    config_model = AzureBatchConfig

    # LLM transforms are non-deterministic by nature
    determinism: Determinism = Determinism.NON_DETERMINISTIC

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Azure Batch LLM transform.

        Args:
            config: Transform configuration dictionary
        """
        super().__init__(config)

        cfg = AzureBatchConfig.from_dict(config, plugin_name=self.name)
        # Retain the typed config so the value-source compliance walker
        # can inspect ``VALUE_SOURCES`` declarations and field values
        # (model / deployment_name) without scraping ``self._field_name``
        # destructured copies. Exposed publicly via ``provider_config``.
        self._config = cfg
        self._initialize_declared_input_fields(cfg)

        # Declare output fields for centralized collision detection.
        self.declared_output_fields = frozenset(get_llm_guaranteed_fields(cfg.response_field))

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
        self.output_schema = _build_augmented_output_schema(
            base_schema_config=schema_config,
            response_field=cfg.response_field,
            schema_name="AzureBatchOutputSchema",
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

        # Azure OpenAI client (lazy init)
        self._client: Any = None
        self._client_lock = Lock()

        # Tier 2: Plugin-internal tracing (Langfuse only for batch API)
        self._tracing_config: TracingConfig | None = parse_tracing_config(cfg.tracing)
        self._tracer = create_langfuse_tracer(
            transform_name=self.name,
            tracing_config=self._tracing_config,
        )
        # TODO: remove bridge — _record_langfuse_batch_job should use self._tracer
        self._tracing_active: bool = isinstance(self._tracer, ActiveLangfuseTracer)
        self._langfuse_client: Any = self._tracer.client if isinstance(self._tracer, ActiveLangfuseTracer) else None

    def on_start(self, ctx: LifecycleContext) -> None:
        """Initialize tracing if configured.

        Called by the engine at pipeline start. Initializes Tier 2 tracing.
        """
        super().on_start(ctx)
        if self._tracing_config is not None:
            self._setup_tracing()

    def _setup_tracing(self) -> None:
        """Initialize Tier 2 tracing based on provider.

        Azure Batch API submits jobs asynchronously - we can only trace at
        the job level, not individual LLM calls. Azure AI auto-instrumentation
        is NOT supported because the OpenAI SDK isn't used for the actual
        LLM inference (that happens in Azure's batch infrastructure).
        """
        if self._tracing_config is None:
            return

        # Validate configuration completeness
        errors = validate_tracing_config(self._tracing_config)
        if errors:
            raise ValueError(f"Tracing configuration errors: {'; '.join(errors)}")

        match self._tracing_config.provider:
            case "azure_ai":
                # Azure AI tracing NOT supported for batch API
                raise ValueError(
                    "Azure AI tracing is not supported for Azure Batch API. "
                    "Azure Batch jobs run asynchronously in Azure infrastructure — "
                    "use Langfuse for job-level tracing instead (provider: langfuse)."
                )
            case "langfuse":
                pass  # Handled by create_langfuse_tracer() in __init__
            case "none":
                pass  # No tracing
            case _:
                raise ValueError(
                    f"Unknown tracing provider '{self._tracing_config.provider}' after validation. Supported: azure_ai, langfuse, none."
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

        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors in our Langfuse integration — crash to surface the bug
        except Exception as e:
            logger.warning("Failed to record Langfuse batch job", error=str(e), exc_info=True)

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
        ctx: TransformContext,
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
        ctx: TransformContext,
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
            # PHASE 2: Resume - batch already submitted
            # BatchCheckpointState is frozen and always has batch_id — no missing-key check needed
            return self._check_batch_status(checkpoint, rows, ctx)

        # PHASE 1: Fresh batch - submit new
        return self._submit_batch(rows, ctx)

    def _get_checkpoint(self, ctx: TransformContext) -> BatchCheckpointState | None:
        """Get checkpoint state from context.

        Args:
            ctx: Plugin context

        Returns:
            Typed checkpoint state or None if no checkpoint
        """
        return ctx.get_checkpoint()

    def _set_checkpoint(self, ctx: TransformContext, state: BatchCheckpointState) -> None:
        """Set checkpoint state.

        Args:
            ctx: Plugin context
            state: Typed checkpoint state to save
        """
        ctx.set_checkpoint(state)

    def _clear_checkpoint(self, ctx: TransformContext) -> None:
        """Clear checkpoint state.

        Args:
            ctx: Plugin context
        """
        ctx.clear_checkpoint()

    def _submit_batch(
        self,
        rows: list[PipelineRow],
        ctx: TransformContext,
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
        row_mapping: dict[str, RowMappingEntry] = {}  # custom_id -> typed entry
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
            row_mapping[custom_id] = RowMappingEntry(
                index=idx,
                variables_hash=rendered.variables_hash,
            )

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
        # Upload file — external API call (Tier 3 boundary)
        start = time.perf_counter()
        try:
            batch_file = client.files.create(
                file=("batch_input.jsonl", file_bytes),
                purpose="batch",
            )
        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors in our request construction — crash to surface the bug
        except Exception as e:
            # External API failure — record and return structured error
            upload_latency = (time.perf_counter() - start) * 1000
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=upload_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=upload_latency,
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
        upload_latency = (time.perf_counter() - start) * 1000

        # Record successful upload in audit trail.
        try:
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=upload_request,
                response_data={"file_id": batch_file.id, "status": batch_file.status},
                latency_ms=upload_latency,
                provider="azure",
            )
        except Exception as exc:
            raise AuditIntegrityError(
                f"Failed to record successful file upload to audit trail "
                f"(file_id={batch_file.id!r}). "
                f"Upload completed but audit record is missing."
            ) from exc

        # Create batch job — external API call (Tier 3 boundary)
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
        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors in our request construction — crash to surface the bug
        except Exception as e:
            # External API failure — record and return structured error
            batch_latency = (time.perf_counter() - start) * 1000
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=batch_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=batch_latency,
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
        batch_latency = (time.perf_counter() - start) * 1000

        # Record successful batch creation in audit trail.
        try:
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=batch_request,
                response_data={"batch_id": batch.id, "status": batch.status},
                latency_ms=batch_latency,
                provider="azure",
            )
        except Exception as exc:
            raise AuditIntegrityError(
                f"Failed to record successful batch creation to audit trail "
                f"(batch_id={batch.id!r}). "
                f"Batch submitted but audit record is missing."
            ) from exc

        # 4. CHECKPOINT immediately after submit
        checkpoint_state = BatchCheckpointState(
            batch_id=batch.id,
            input_file_id=batch_file.id,
            row_mapping=row_mapping,
            template_errors=template_errors,
            submitted_at=datetime.now(UTC).isoformat(),
            row_count=len(rows),
            requests=requests_by_id,  # BUG-AZURE-01: Store original requests for audit recording
        )
        self._set_checkpoint(ctx, checkpoint_state)

        # 5. Raise BatchPendingError - engine handles retry scheduling
        # Include checkpoint and node_id so caller can persist and restore
        raise BatchPendingError(
            batch.id,
            "submitted",
            check_after_seconds=self._poll_interval,
            checkpoint=checkpoint_state,
            node_id=self.node_id,
        )

    def _record_per_row_failure_calls(
        self,
        checkpoint: BatchCheckpointState,
        ctx: TransformContext,
        terminal_status: str,
        error_detail: str | None = None,
        *,
        reason_prefix: str = "batch_",
    ) -> None:
        """Record per-row LLM call records for terminal batch failure paths.

        When a batch fails/cancels/expires/times-out, the per-row LLM calls are
        never recorded because _download_results() (the only place that records them)
        is only called on the "completed" path. This method ensures audit completeness
        by emitting a CallType.LLM error record for each submitted request before
        the checkpoint is cleared.

        Args:
            checkpoint: Typed checkpoint state with requests and row_mapping.
            ctx: Plugin context for record_call.
            terminal_status: The batch terminal status (failed, cancelled, expired, timeout).
                Prefixed with ``reason_prefix`` to produce the audit reason
                (e.g. ``reason_prefix="batch_", terminal_status="timeout"`` →
                ``"batch_timeout"``).
            error_detail: Optional extra error detail string.
            reason_prefix: Prefix for the structured error reason.
        """
        if not checkpoint.requests and not checkpoint.row_mapping:
            return  # No requests were submitted (e.g., all templates failed)
        if bool(checkpoint.requests) != bool(checkpoint.row_mapping):
            raise RuntimeError(
                f"Checkpoint invariant violation: requests and row_mapping must both be "
                f"empty or both be populated, but got "
                f"{len(checkpoint.requests)} requests and {len(checkpoint.row_mapping)} row_mapping entries "
                f"for batch {checkpoint.batch_id!r}. This indicates checkpoint corruption."
            )

        for custom_id, original_request in checkpoint.requests.items():
            mapping_entry = checkpoint.row_mapping[custom_id]
            row_index = mapping_entry.index
            error_info: dict[str, Any] = {
                "reason": f"{reason_prefix}{terminal_status}",
                "batch_id": checkpoint.batch_id,
                "custom_id": custom_id,
            }
            if error_detail:
                error_info["detail"] = error_detail

            # Audit completeness is non-negotiable (Tier 1 trust model). If we
            # cannot record the per-row call, the audit trail is compromised and
            # partial progress is worthless — crash immediately.
            try:
                ctx.record_call(
                    call_type=CallType.LLM,
                    status=CallStatus.ERROR,
                    request_data={
                        "custom_id": custom_id,
                        "row_index": row_index,
                        "variables_hash": mapping_entry.variables_hash,
                        **original_request,
                    },
                    response_data=None,
                    error=error_info,
                    provider="azure",
                )
            except Exception as exc:
                raise AuditIntegrityError(
                    f"Failed to record per-row LLM call audit for batch {checkpoint.batch_id!r}, "
                    f"custom_id={custom_id!r}, row_index={row_index}. "
                    f"Audit trail integrity compromised."
                ) from exc

    def _check_batch_status(
        self,
        checkpoint: BatchCheckpointState,
        rows: list[PipelineRow],
        ctx: TransformContext,
    ) -> TransformResult:
        """Check batch status and complete if ready.

        Args:
            checkpoint: Typed checkpoint state with batch_id
            rows: Original rows (needed for output assembly)
            ctx: Plugin context

        Returns:
            TransformResult with results if complete

        Raises:
            BatchPendingError: If batch still in progress
        """
        batch_id = checkpoint.batch_id
        client = self._get_client()

        # Check batch status (with audit recording)
        retrieve_request = {
            "operation": "batches.retrieve",
            "batch_id": batch_id,
        }
        # Retrieve batch status — external API call (Tier 3 boundary)
        start = time.perf_counter()
        try:
            batch = client.batches.retrieve(batch_id)
        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors — crash to surface the bug
        except Exception as e:
            # External API failure after submission — keep the checkpoint and
            # resume via the pending control-flow path instead of finalizing
            # the batch locally as FAILED.
            retrieve_latency = (time.perf_counter() - start) * 1000
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=retrieve_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=retrieve_latency,
                provider="azure",
            )
            raise BatchPendingError(
                batch_id,
                "status_check_retry",
                check_after_seconds=self._poll_interval,
                checkpoint=checkpoint,
                node_id=self.node_id,
            ) from e
        retrieve_latency = (time.perf_counter() - start) * 1000

        # Record successful retrieve in audit trail. Tier 3 boundary: SDK response
        # shape varies across versions (output_file_id/error_file_id may be absent
        # on pending batches or older SDK versions).
        try:
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=retrieve_request,
                response_data={
                    "batch_id": batch.id,
                    "status": batch.status,
                    "output_file_id": getattr(batch, "output_file_id", None),
                    "error_file_id": getattr(batch, "error_file_id", None),
                },
                latency_ms=retrieve_latency,
                provider="azure",
            )
        except Exception as exc:
            raise AuditIntegrityError(
                f"Failed to record successful batch retrieve to audit trail "
                f"(batch_id={batch.id!r}). "
                f"Retrieve completed but audit record is missing."
            ) from exc

        if batch.status == "completed":
            # Calculate latency from submission to completion
            submitted_at = datetime.fromisoformat(checkpoint.submitted_at)
            latency_ms = (datetime.now(UTC) - submitted_at).total_seconds() * 1000

            # Record to Langfuse (job-level tracing)
            self._record_langfuse_batch_job(
                batch_id=batch_id,
                row_count=checkpoint.row_count,
                latency_ms=latency_ms,
                status="completed",
            )

            # Download results and assemble output
            return self._download_results(batch, checkpoint, rows, ctx)

        elif batch.status == "failed":
            # Batch failed - record per-row calls, then clear checkpoint and return error

            error_info: TransformErrorReason = {
                "reason": "batch_failed",
                "batch_id": batch_id,
            }
            error_message = None
            # Tier 3 boundary: batch.errors is from Azure SDK — the wrapper
            # object is always truthy when present, so .data must be checked
            # separately for None or empty list.  Error entries inside .data
            # are also Tier 3 — probe SDK shape once before bulk access so
            # that AttributeError from our own typos isn't silently caught.
            if batch.errors and batch.errors.data:
                try:
                    probe = batch.errors.data[0]
                    probe.message  # noqa: B018 — shape probe, not dead code
                    probe.code  # noqa: B018
                except AttributeError:
                    error_info["errors"] = [{"message": "batch error details unparseable", "error_type": "unknown"}]
                else:
                    error_info["errors"] = [{"message": e.message, "error_type": e.code} for e in batch.errors.data]
                    error_message = "; ".join(e.message for e in batch.errors.data)

            # Extract checkpoint values before clearing
            submitted_at = datetime.fromisoformat(checkpoint.submitted_at)
            row_count = checkpoint.row_count

            # Record per-row LLM calls BEFORE clearing checkpoint for audit completeness
            self._record_per_row_failure_calls(
                checkpoint,
                ctx,
                terminal_status="failed",
                error_detail=error_message,
            )
            self._clear_checkpoint(ctx)

            # Calculate latency for failed batch
            latency_ms = (datetime.now(UTC) - submitted_at).total_seconds() * 1000

            # Record failure to Langfuse
            self._record_langfuse_batch_job(
                batch_id=batch_id,
                row_count=row_count,
                latency_ms=latency_ms,
                status="failed",
                error=error_message,
            )

            return TransformResult.error(error_info)

        elif batch.status == "cancelled":
            # Extract checkpoint values before clearing
            submitted_at = datetime.fromisoformat(checkpoint.submitted_at)
            row_count = checkpoint.row_count

            # Batch was cancelled - record per-row calls, then clear checkpoint and return error
            self._record_per_row_failure_calls(checkpoint, ctx, terminal_status="cancelled")
            self._clear_checkpoint(ctx)

            # Calculate latency for cancelled batch
            latency_ms = (datetime.now(UTC) - submitted_at).total_seconds() * 1000

            # Record cancellation to Langfuse
            self._record_langfuse_batch_job(
                batch_id=batch_id,
                row_count=row_count,
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
            # Batch expired (exceeded 24h) - record per-row calls, clear checkpoint and return error
            self._record_per_row_failure_calls(checkpoint, ctx, terminal_status="expired")
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
            submitted_at = datetime.fromisoformat(checkpoint.submitted_at)
            elapsed_hours = (datetime.now(UTC) - submitted_at).total_seconds() / 3600

            if elapsed_hours > self._max_wait_hours:
                self._record_per_row_failure_calls(checkpoint, ctx, terminal_status="timeout")
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
        checkpoint: BatchCheckpointState,
        rows: list[PipelineRow],
        ctx: TransformContext,
    ) -> TransformResult:
        """Download batch results and assemble output rows.

        Args:
            batch: Completed Azure batch object
            checkpoint: Typed checkpoint state with row mapping
            rows: Original input rows (list[PipelineRow])
            ctx: Plugin context

        Returns:
            TransformResult with all processed rows
        """
        client = self._get_client()
        # BatchCheckpointState is frozen — row_mapping and template_errors always present
        row_mapping = checkpoint.row_mapping
        template_errors = checkpoint.template_errors

        # Download output file (with audit recording)
        output_file_id = batch.output_file_id
        download_request = {
            "operation": "files.content",
            "file_id": output_file_id,
        }
        # Download output file — external API call (Tier 3 boundary)
        start = time.perf_counter()
        try:
            output_content = client.files.content(output_file_id)
            output_text = output_content.text
        except (TypeError, AttributeError, KeyError, NameError):
            raise  # Programming errors — crash to surface the bug
        except Exception as e:
            # Azure already completed the batch; preserve the checkpoint and
            # retry the download instead of converting this to a terminal batch
            # failure locally.
            download_latency = (time.perf_counter() - start) * 1000
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data=download_request,
                response_data={"error": str(e), "error_type": type(e).__name__},
                latency_ms=download_latency,
                provider="azure",
            )
            raise BatchPendingError(
                checkpoint.batch_id,
                "download_retry",
                check_after_seconds=self._poll_interval,
                checkpoint=checkpoint,
                node_id=self.node_id,
            ) from e
        download_latency = (time.perf_counter() - start) * 1000

        # Record successful download in audit trail.
        output_hash = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        try:
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.SUCCESS,
                request_data=download_request,
                response_data={
                    "file_id": output_file_id,
                    "content_sha256": output_hash,
                    "content_length": len(output_text),
                },
                latency_ms=download_latency,
                provider="azure",
            )
        except Exception as exc:
            raise AuditIntegrityError(
                f"Failed to record successful output file download to audit trail "
                f"(file_id={output_file_id!r}). "
                f"Download completed but audit record is missing."
            ) from exc

        # Download error file if present (partial batch failures)
        # Azure Batch API puts per-request errors in a separate error_file_id
        # Tier 3 boundary: SDK must expose this attribute — None = no errors,
        # missing attribute = SDK version mismatch (crash, don't silently skip)
        error_file_download_failed = False
        try:
            error_file_id = batch.error_file_id
        except AttributeError:
            raise RuntimeError(
                f"Azure batch object missing 'error_file_id' attribute. "
                f"Ensure Azure OpenAI SDK version >= 1.14.0 supports batch error files. "
                f"batch_id={batch.id}"
            ) from None
        if error_file_id is not None:
            error_download_request = {
                "operation": "files.content",
                "file_id": error_file_id,
                "file_type": "error",
            }
            # Download error file — external API call (Tier 3 boundary)
            start = time.perf_counter()
            try:
                error_content = client.files.content(error_file_id)
                error_text = error_content.text
            except (TypeError, AttributeError, KeyError, NameError):
                raise  # Programming errors — crash to surface the bug
            except Exception as e:
                # Error file download failed — record in audit trail and flag for downstream
                # The output file was already downloaded successfully, but per-row error
                # details from the error file are now permanently lost. Rows that had
                # Azure-side errors will be misclassified as result_not_found without this flag.
                error_file_download_failed = True
                error_latency = (time.perf_counter() - start) * 1000
                ctx.record_call(
                    call_type=CallType.HTTP,
                    status=CallStatus.ERROR,
                    request_data=error_download_request,
                    response_data={"error": str(e), "error_type": type(e).__name__},
                    latency_ms=error_latency,
                    provider="azure",
                )
                error_text = None
            else:
                # Record successful error file download in audit trail.
                error_latency = (time.perf_counter() - start) * 1000
                error_hash = hashlib.sha256(error_text.encode("utf-8")).hexdigest()
                try:
                    ctx.record_call(
                        call_type=CallType.HTTP,
                        status=CallStatus.SUCCESS,
                        request_data=error_download_request,
                        response_data={
                            "file_id": error_file_id,
                            "content_sha256": error_hash,
                            "content_length": len(error_text),
                        },
                        latency_ms=error_latency,
                        provider="azure",
                    )
                except Exception as exc:
                    raise AuditIntegrityError(
                        f"Failed to record successful error file download to audit trail "
                        f"(file_id={error_file_id!r}). "
                        f"Download completed but audit record is missing."
                    ) from exc
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
                if not isinstance(response["body"], dict):
                    malformed_lines.append(f"Line {line_num}: 'response.body' is not a dict, got {type(response['body']).__name__}")
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
            self._record_per_row_failure_calls(
                checkpoint,
                ctx,
                "all_output_lines_malformed",
                error_detail=f"{len(malformed_lines)} malformed output line(s)",
                reason_prefix="",
            )
            self._clear_checkpoint(ctx)
            return TransformResult.error(
                {
                    "reason": "all_output_lines_malformed",
                    "malformed_count": len(malformed_lines),
                    "errors": list(malformed_lines[:10]),  # First 10 errors for diagnosis
                }
            )

        # Partial malformed: some lines parsed, some didn't — record the malformed ones
        # so the audit trail shows WHY certain rows have no results. Without this,
        # malformed lines silently vanish and their rows get misclassified as result_not_found.
        if malformed_lines and results_by_id:
            ctx.record_call(
                call_type=CallType.HTTP,
                status=CallStatus.ERROR,
                request_data={
                    "operation": "parse_batch_jsonl",
                    "batch_id": batch.id,
                },
                response_data={
                    "malformed_count": len(malformed_lines),
                    "parsed_count": len(results_by_id),
                    "total_lines": len(malformed_lines) + len(results_by_id),
                    "errors": malformed_lines[:20],
                },
                provider="azure",
            )

        # Assemble output rows in original order
        output_rows: list[dict[str, Any]] = []
        row_errors: list[RowErrorEntry] = []
        finish_reason_counts: Counter[str] = Counter()

        # Track which rows had template errors (excluded from batch)
        template_error_indices = {idx for idx, _ in template_errors}

        # Build reverse mapping once (O(n) instead of O(n^2) lookup per row)
        idx_to_custom_id: dict[int, str] = {info.index: cid for cid, info in row_mapping.items()}

        for idx, row in enumerate(rows):
            if idx in template_error_indices:
                # Template-render failures are quarantined metadata, not emitted
                # child rows. Emitting them via success_multi() would route them
                # downstream despite the parent token being quarantined.
                error_msg = next((err for i, err in template_errors if i == idx), "Unknown error")
                row_errors.append(
                    {
                        "row_index": idx,
                        "reason": "template_rendering_failed",
                        "error": {
                            "reason": "template_rendering_failed",
                            "error": error_msg,
                        },
                    }
                )
                continue

            # Find result by custom_id using pre-built reverse mapping
            if idx not in idx_to_custom_id:
                raise RuntimeError(f"Checkpoint row_mapping missing entry for row index {idx} in AzureBatchLLMTransform.")
            custom_id = idx_to_custom_id[idx]

            if custom_id not in results_by_id:
                # Row's custom_id not found in parsed results. Two distinct causes:
                # 1. Error file download failed — row's error details were in the error
                #    file which couldn't be downloaded (error_details_unavailable)
                # 2. Malformed JSONL — row's line was unparseable, custom_id never extracted
                # 3. True absence — Azure Batch API edge case, no output produced
                # The error_file_download_failed flag distinguishes cause 1.
                if error_file_download_failed:
                    error_reason = "error_details_unavailable"
                    error_detail = (
                        "Row may have had per-row errors from Azure, but the error file "
                        "download failed. Error details are permanently lost."
                    )
                else:
                    error_reason = "result_not_found"
                    error_detail = None

                row_errors.append({"row_index": idx, "reason": error_reason})

                # Record Call for audit trail completeness - request WAS made but no response
                # Without this, explain(token_id) would show incomplete lineage
                # Access directly from checkpoint (Tier 1 data - we wrote it)
                # Audit completeness is non-negotiable (Tier 1 trust model). If we
                # cannot record the per-row call, the audit trail is compromised.
                original_request = checkpoint.requests[custom_id]
                error_payload: dict[str, Any] = {
                    "reason": error_reason,
                    "custom_id": custom_id,
                }
                if error_detail is not None:
                    error_payload["detail"] = error_detail
                try:
                    ctx.record_call(
                        call_type=CallType.LLM,
                        status=CallStatus.ERROR,
                        request_data={
                            "custom_id": custom_id,
                            "row_index": idx,
                            "variables_hash": row_mapping[custom_id].variables_hash,
                            **original_request,
                        },
                        response_data=None,
                        error=error_payload,
                        provider="azure",
                    )
                except Exception as exc:
                    raise AuditIntegrityError(
                        f"Failed to record per-row LLM call audit for batch {checkpoint.batch_id!r}, "
                        f"custom_id={custom_id!r}, row_index={idx}. "
                        f"Audit trail integrity compromised."
                    ) from exc
                continue

            result = results_by_id[custom_id]

            if "error" in result:
                # API error for this row. Keep the failure in metadata only so
                # the engine does not expand it into a downstream child token.
                row_errors.append({"row_index": idx, "reason": "api_error", "error": result["error"]})
            else:
                # Success - extract response
                response = result["response"]
                body = response["body"]

                # Tier 3 boundary validation: Azure API response structure
                # Validate expected fields exist with correct types before accessing
                choices = body.get("choices")
                if not isinstance(choices, list) or len(choices) == 0:
                    row_errors.append(
                        {
                            "row_index": idx,
                            "reason": "no_choices_in_response",
                            "error": "Azure API response missing 'choices' array",
                        }
                    )
                    continue

                first_choice = choices[0]
                if not isinstance(first_choice, dict):
                    row_errors.append(
                        {
                            "row_index": idx,
                            "reason": "invalid_choice_structure",
                            "error": f"choices[0] is not a dict, got {type(first_choice).__name__}",
                        }
                    )
                    continue

                message = first_choice.get("message")
                if not isinstance(message, dict):
                    row_errors.append(
                        {
                            "row_index": idx,
                            "reason": "invalid_message_structure",
                            "error": f"choices[0].message is not a dict, got {type(message).__name__}",
                        }
                    )
                    continue

                # Boundary validation passed — check content at Tier 3 boundary
                content = message.get("content")
                if content is None:
                    # Null content = content filtered by provider (Tier 3 boundary)
                    # Matches the openrouter_batch pattern.
                    row_errors.append({"row_index": idx, "reason": "content_filtered"})
                    continue

                # Extract finish_reason for audit trail (Tier 3 boundary)
                raw_finish_reason = first_choice.get("finish_reason")

                # Usage is optional in Azure API — from_dict handles None gracefully
                usage = TokenUsage.from_dict(body.get("usage"))

                # Field collision check already done in _submit_batch() before
                # submitting the batch — no need to re-check here.

                fr_key = raw_finish_reason or "absent"
                finish_reason_counts[fr_key] += 1

                output_row = row.to_dict()
                output_row[self._response_field] = content

                populate_llm_operational_fields(
                    output_row,
                    self._response_field,
                    usage=usage,
                    # Azure batch responses may omit 'model' — fall back to the
                    # deployment name we sent in the request (known ground truth).
                    model=body.get("model") or self._deployment_name,
                )

                output_rows.append(output_row)

        # Record per-row LLM calls against the batch's state
        # Uses existing state_id from context (set by AggregationExecutor)
        # Note: checkpoint.requests is Tier 1 data (we wrote it) - typed access
        requests_data = checkpoint.requests

        for custom_id, result in results_by_id.items():
            original_request = requests_data[custom_id]
            mapping_entry = row_mapping[custom_id]
            row_index = mapping_entry.index

            # Determine call status from result (Tier 2 - validated at boundary)
            if "error" in result:
                call_status = CallStatus.ERROR
                response_data = None
                error_data = {"reason": "api_error", "error": result["error"]}
            else:
                call_status = CallStatus.SUCCESS
                # response.body guaranteed by boundary validation
                response_data = result["response"]["body"]
                error_data = None

            # Record LLM call with custom_id for token mapping
            # Audit completeness is non-negotiable (Tier 1 trust model). If we
            # cannot record the per-row call, the audit trail is compromised and
            # partial progress is worthless — crash immediately.
            try:
                ctx.record_call(
                    call_type=CallType.LLM,
                    status=call_status,
                    request_data={
                        "custom_id": custom_id,
                        "row_index": row_index,
                        "variables_hash": mapping_entry.variables_hash,
                        **original_request,
                    },
                    response_data=response_data,
                    error=error_data,
                    provider="azure",
                )
            except Exception as exc:
                raise AuditIntegrityError(
                    f"Failed to record per-row LLM call audit for batch {checkpoint.batch_id!r}, "
                    f"custom_id={custom_id!r}, row_index={row_index}. "
                    f"Audit trail integrity compromised."
                ) from exc

        # The batch is terminal at this point regardless of whether any valid
        # rows survived. Clear the checkpoint before returning.
        self._clear_checkpoint(ctx)

        if not output_rows:
            return TransformResult.error(
                {
                    "reason": "all_rows_failed",
                    "row_errors": row_errors,
                }
            )

        # Create OBSERVED contract from union of ALL output row keys (not just first)
        # Error rows may have extra fields (e.g. _error) that the first row lacks
        # Infer python_type from first non-None value seen per key across all rows
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
        output_contract = self._align_output_contract(output_contract)

        # Batch-level template provenance — shared across all rows in the batch.
        # Per-row request/response hashes are in the calls table via record_call().
        batch_audit = build_llm_audit_metadata(
            self._response_field,
            template_hash=self._template.template_hash,
            variables_hash=None,  # Batch-level: per-row hashes are in calls table request_data
            template_source=self._template.template_source,
            lookup_hash=self._template.lookup_hash,
            lookup_source=self._template.lookup_source,
            system_prompt_source=self._system_prompt_source,
        )

        # Collect all quarantined row indices for engine visibility.
        # Without quarantined_indices in metadata, the engine marks ALL rows as
        # CONSUMED_IN_BATCH — error rows silently pass as successfully processed.
        quarantined_indices = sorted({e["row_index"] for e in row_errors})

        success_reason: TransformSuccessReason = {
            "action": "enriched",
            "fields_added": [self._response_field],
            "metadata": {
                "batch_size": len(rows),
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

    @property
    def provider_config(self) -> AzureBatchConfig:
        """Read-only accessor for the typed batch config.

        Mirrors :attr:`elspeth.plugins.transforms.llm.transform.LLMTransform.provider_config`
        so the value-source compliance walker can inspect the
        ``VALUE_SOURCES`` declaration and post-validation field values
        without reaching into ``self._config`` private state. Named
        ``provider_config`` (not ``config``) to avoid colliding with
        :attr:`BaseTransform.config` (the raw ``dict[str, Any]``).
        """
        return self._config

    def close(self) -> None:
        """Release resources and flush tracing."""
        self._tracer.flush()
        if self._client is not None:
            self._client.close()
            self._client = None
        self._langfuse_client = None


# Register opt-in for value-source compliance: the typed Pydantic config
# (AzureBatchConfig) is exposed via the public ``provider_config``
# property. Mirrors the non-batch registration at transform.py:1480 so
# the walker (engine/orchestrator/preflight.py) treats batch and
# non-batch Azure transforms identically.
register_value_source_plugin(AzureBatchLLMTransform, config_attr="provider_config")
