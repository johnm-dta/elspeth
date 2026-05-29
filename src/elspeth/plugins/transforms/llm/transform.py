"""Unified LLM transform with strategy pattern.

LLMTransform dispatches to SingleQueryStrategy or MultiQueryStrategy
based on whether queries are configured. Provider dispatch (Azure,
OpenRouter) is handled via _PROVIDERS registry.

Architecture:
    LLMTransform (BatchTransformMixin)
        ├── SingleQueryStrategy — direct template render, raw content output
        └── MultiQueryStrategy — mapped context, JSON parsing, field extraction

    Provider instantiation is deferred to on_start() when recorder/telemetry
    become available. __init__ stores provider_cls + config for later use.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, Annotated, Any, Protocol, cast

import structlog
from pydantic import BaseModel, TypeAdapter
from pydantic import Field as PydanticField

from elspeth.contracts import Determinism, TransformErrorReason, TransformResult, propagate_contract
from elspeth.contracts.audit_protocols import PluginAuditWriter
from elspeth.contracts.contexts import LifecycleContext, TransformContext
from elspeth.contracts.errors import FrameworkBugError, RuntimePreflightFailedError
from elspeth.contracts.freeze import freeze_fields
from elspeth.contracts.plugin_assistance import PluginAssistance
from elspeth.contracts.schema_contract import FieldContract, PipelineRow, SchemaContract
from elspeth.contracts.token_usage import TokenUsage
from elspeth.contracts.value_source import register_value_source_plugin
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.batching import BatchTransformMixin, OutputPort
from elspeth.plugins.infrastructure.clients.llm import ContextLengthError, LLMClientError
from elspeth.plugins.infrastructure.pooling import PooledExecutor, RowContext
from elspeth.plugins.infrastructure.schema_factory import create_schema_from_config
from elspeth.plugins.infrastructure.telemetry import make_warn_telemetry_before_start
from elspeth.plugins.infrastructure.templates import TemplateError
from elspeth.plugins.transforms.llm import (
    _OUTPUT_FIELD_TYPE_TO_SCHEMA,
    _build_augmented_output_schema,
    _build_llm_output_schema_config,
    _build_multi_query_output_schema,
    _FieldType,
    build_llm_audit_metadata,
    get_llm_guaranteed_fields,
    populate_llm_operational_fields,
)
from elspeth.plugins.transforms.llm.base import LLMConfig
from elspeth.plugins.transforms.llm.langfuse import LangfuseTracer, create_langfuse_tracer
from elspeth.plugins.transforms.llm.multi_query import QuerySpec, ResponseFormat, resolve_queries
from elspeth.plugins.transforms.llm.provider import (
    FinishReason,
    LLMProvider,
    LLMQueryResult,
    ParsedFinishReason,
    UnrecognizedFinishReason,
)
from elspeth.plugins.transforms.llm.providers.azure import AzureLLMProvider, AzureOpenAIConfig, _configure_azure_monitor
from elspeth.plugins.transforms.llm.providers.openrouter import OpenRouterConfig, OpenRouterLLMProvider
from elspeth.plugins.transforms.llm.templates import PromptTemplate
from elspeth.plugins.transforms.llm.tracing import AzureAITracingConfig, TracingConfig, parse_tracing_config
from elspeth.plugins.transforms.llm.validation import reject_nonfinite_constant, strip_markdown_fences, validate_field_value

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from elspeth.contracts.plugin_semantics import OutputSemanticDeclaration


_warn_telemetry_before_start = make_warn_telemetry_before_start(logger)


_FINISH_REASON_ERRORS: dict[FinishReason, tuple[str, str]] = {
    FinishReason.LENGTH: ("response_truncated", "Response truncated (finish_reason=length)"),
    FinishReason.CONTENT_FILTER: ("content_filtered", "Response blocked by provider content filter"),
}


@dataclass(frozen=True, slots=True)
class _FinishReasonError:
    """Bundles the transform result and tracer message for a terminal finish reason."""

    result: TransformResult
    error_message: str


def _serialize_finish_reason(finish_reason: ParsedFinishReason) -> str | None:
    """Serialize finish_reason for audit metadata and error reporting.

    Single source of truth for converting ParsedFinishReason to its string
    representation. Used by both _finish_reason_error (error path) and
    success metadata recording (audit path) to eliminate parallel isinstance
    dispatch chains that would need synchronized updates.

    Returns a string for known enum values, the raw string for unrecognized
    values, and None when the provider couldn't extract it. The audit trail
    must distinguish these three cases.
    """
    if finish_reason is None:
        return None
    if isinstance(finish_reason, FinishReason):
        return finish_reason.value
    if isinstance(finish_reason, UnrecognizedFinishReason):
        return finish_reason.raw
    return str(finish_reason)  # type: ignore[unreachable]  # pragma: no cover — exhaustive, but future-proof


def _shutdown_event_is_set(shutdown_event: object | None) -> bool:
    """Return True only for concrete cancellation signals, not arbitrary mocks."""
    is_set = getattr(shutdown_event, "is_set", None)
    return callable(is_set) and is_set() is True


def _shutdown_requested_result(
    *,
    error: str = "Run cancellation requested before LLM provider call",
    query_name: str | None = None,
    query_index: int | None = None,
) -> TransformResult:
    reason: dict[str, Any] = {"reason": "shutdown_requested", "error": error}
    if query_name is not None:
        reason["query_name"] = query_name
    if query_index is not None:
        reason["query_index"] = query_index
    return TransformResult.error(cast(TransformErrorReason, reason), retryable=False)


def _finish_reason_error(
    finish_reason: ParsedFinishReason,
    *,
    query_name: str | None = None,
    query_index: int | None = None,
    content_length: int | None = None,
) -> _FinishReasonError | None:
    """Fail closed on non-STOP finish reasons.

    Only explicit STOP is allowlisted.  Absent finish_reason (None) is
    accepted with a structured warning (Drifting Goals intervention — see
    below). Known-bad reasons get specific error messages, unknown/unrecognized
    reasons get a generic rejection. This ensures new provider finish reasons
    are never silently treated as success.
    """

    def _build_reason(**base_fields: str) -> dict[str, Any]:
        reason: dict[str, Any] = dict(base_fields)
        if query_name is not None:
            reason["query_name"] = query_name
        if query_index is not None:
            reason["query_index"] = query_index
        if content_length is not None:
            reason["content_length"] = content_length
        return reason

    # Allowlist: explicit STOP is a known-good completion.
    if finish_reason == FinishReason.STOP:
        return None

    # Absent finish_reason (None) is a valid response shape for some providers
    # (e.g. Azure SDK omits raw_response or choices in certain configurations).
    # This is provider-normal behavior, not a defect. The provider already
    # validated content is non-empty via LLMQueryResult, and logged a warning
    # about "truncation undetectable".
    #
    # Callers record finish_reason in success_reason.metadata so the audit
    # trail distinguishes None (absent) from STOP (confirmed completion).
    # This is queryable via MCP diagnose() for operational visibility.
    if finish_reason is None:
        return None

    # Known-bad reasons with specific error messages.
    if isinstance(finish_reason, FinishReason):
        entry = _FINISH_REASON_ERRORS.get(finish_reason)
        if entry is not None:
            reason_key, error_message = entry
            return _FinishReasonError(
                result=TransformResult.error(
                    cast(
                        TransformErrorReason,
                        _build_reason(reason=reason_key, finish_reason=finish_reason.value),
                    ),
                    retryable=False,
                ),
                error_message=error_message,
            )
        # entry is None: this FinishReason is not in the error dict but is also
        # not STOP — fall through to the catch-all so it is rejected.

    # Catch-all: any finish reason not explicitly allowlisted (including
    # known enum members not in STOP or error dict, and unrecognized values)
    # is an error. Uses _serialize_finish_reason as the single source of truth
    # for string conversion. None was handled above; raw_value is always str.
    raw_value = _serialize_finish_reason(finish_reason)
    assert raw_value is not None, "finish_reason=None was handled above — unreachable"
    return _FinishReasonError(
        result=TransformResult.error(
            cast(
                TransformErrorReason,
                _build_reason(reason="unexpected_finish_reason", finish_reason=raw_value),
            ),
            retryable=False,
        ),
        error_message=f"Unexpected finish reason: {raw_value}",
    )


# ---------------------------------------------------------------------------
# Provider registry — single source of truth for both config parsing and
# provider instantiation. Eliminates the sync failure mode of two dispatch tables.
# ---------------------------------------------------------------------------

# NOTE: type[LLMProvider] won't work here — mypy doesn't support type[Protocol]
# for structural subtyping. The concrete classes (AzureLLMProvider,
# OpenRouterLLMProvider) are verified against LLMProvider by mypy at their
# definition sites. The provider class is stored here for documentation only —
# actual construction uses isinstance narrowing in _create_provider().
_PROVIDERS: dict[str, tuple[type[LLMConfig], type]] = {
    "azure": (AzureOpenAIConfig, AzureLLMProvider),
    "openrouter": (OpenRouterConfig, OpenRouterLLMProvider),
}


# ---------------------------------------------------------------------------
# Strategy protocol and implementations
# ---------------------------------------------------------------------------


class QueryStrategy(Protocol):
    """What LLMTransform._process_row delegates to."""

    def execute(
        self,
        row: PipelineRow,
        ctx: TransformContext,
        *,
        provider: LLMProvider,
        tracer: LangfuseTracer,
    ) -> TransformResult: ...


@dataclass(frozen=True, slots=True)
class SingleQueryStrategy:
    """Direct template render → LLM call → raw content output."""

    template: PromptTemplate
    system_prompt: str | None
    system_prompt_source: str | None
    model: str
    temperature: float
    max_tokens: int | None
    response_field: str
    align_output_contract: Callable[[SchemaContract], SchemaContract]

    def execute(
        self,
        row: PipelineRow,
        ctx: TransformContext,
        *,
        provider: LLMProvider,
        tracer: LangfuseTracer,
    ) -> TransformResult:
        """Execute single LLM query and build output row."""
        state_id = ctx.state_id
        if state_id is None:
            raise RuntimeError("LLMTransform requires state_id")
        if ctx.token is None:
            raise RuntimeError("LLMTransform requires ctx.token")
        token_id = ctx.token.token_id
        shutdown_event = cast("threading.Event | None", getattr(ctx, "shutdown_event", None))

        # 1. Render template (THEIR DATA — wrap)
        try:
            rendered = self.template.render_with_metadata(row, contract=row.contract)
        except TemplateError as e:
            error_reason: TransformErrorReason = {
                "reason": "template_rendering_failed",
                "error": str(e),
                "template_hash": self.template.template_hash,
            }
            if self.template.template_source:
                error_reason["template_file_path"] = self.template.template_source
            return TransformResult.error(error_reason)

        # 2. Build messages
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        if _shutdown_event_is_set(shutdown_event):
            return _shutdown_requested_result()

        # 3. Call provider (EXTERNAL — errors classified by provider)
        start_time = time.monotonic()
        try:
            result = provider.execute_query(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                state_id=state_id,
                token_id=token_id,
            )
        except ContextLengthError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            tracer.record_error(
                token_id=token_id,
                query_name="single",
                prompt=rendered.prompt,
                error_message=str(e),
                model=self.model,
                latency_ms=latency_ms,
            )
            return TransformResult.error(
                {"reason": "context_length_exceeded", "error": str(e)},
                retryable=False,
            )
        except LLMClientError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            tracer.record_error(
                token_id=token_id,
                query_name="single",
                prompt=rendered.prompt,
                error_message=str(e),
                model=self.model,
                latency_ms=latency_ms,
            )
            if e.retryable:
                raise
            return TransformResult.error(
                {"reason": "llm_call_failed", "error": str(e)},
                retryable=False,
            )

        latency_ms = (time.monotonic() - start_time) * 1000

        # 4. Fail closed on provider-signaled terminal finish reasons.
        finish_reason_error = _finish_reason_error(
            result.finish_reason,
            content_length=len(result.content),
        )
        if finish_reason_error is not None:
            tracer.record_error(
                token_id=token_id,
                query_name="single",
                prompt=rendered.prompt,
                error_message=finish_reason_error.error_message,
                model=self.model,
                latency_ms=latency_ms,
            )
            return finish_reason_error.result

        # 5. Strip markdown fences
        content = strip_markdown_fences(result.content)

        # Record success in tracer
        tracer.record_success(
            token_id=token_id,
            query_name="single",
            prompt=rendered.prompt,
            response_content=content,
            model=self.model,
            usage=result.usage,
            latency_ms=latency_ms,
        )

        # 6. Build output row — operational fields only
        output = row.to_dict()
        output[self.response_field] = content
        populate_llm_operational_fields(
            output,
            self.response_field,
            usage=result.usage,
            model=result.model,
        )

        # 7. Build audit metadata (goes to success_reason, not the row)
        audit_metadata = build_llm_audit_metadata(
            self.response_field,
            template_hash=rendered.template_hash,
            variables_hash=rendered.variables_hash,
            template_source=rendered.template_source,
            lookup_hash=rendered.lookup_hash,
            lookup_source=rendered.lookup_source,
            system_prompt_source=self.system_prompt_source,
        )

        # 8. Propagate contract
        output_contract = propagate_contract(
            input_contract=row.contract,
            output_row=output,
            transform_adds_fields=True,
        )
        output_contract = self.align_output_contract(output_contract)

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "enriched",
                "fields_added": [self.response_field],
                "metadata": {
                    "model": result.model,
                    "finish_reason": _serialize_finish_reason(result.finish_reason),
                    **result.usage.to_dict(),
                    **audit_metadata,
                },
            },
        )


@dataclass(frozen=True, slots=True)
class MultiQueryStrategy:
    """Execute multiple queries per row with atomic failure semantics.

    Supports two execution modes:
    - Sequential (executor=None, pool_size=1): queries run one-by-one
    - Parallel (executor set, pool_size>1): queries run via PooledExecutor
      with AIMD throttle backoff on rate limits

    In parallel mode, the PooledExecutor retries individual queries with
    adaptive backoff, avoiding wasteful full-row retries that discard
    successful query results.
    """

    query_specs: Sequence[QuerySpec]
    template: PromptTemplate
    system_prompt: str | None
    system_prompt_source: str | None
    model: str
    temperature: float
    max_tokens: int | None
    response_field: str
    align_output_contract: Callable[[SchemaContract], SchemaContract]
    align_output_row_contract: Callable[[PipelineRow], PipelineRow]
    executor: PooledExecutor | None = None
    _query_templates: Mapping[str, PromptTemplate] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_specs, tuple):
            object.__setattr__(self, "query_specs", tuple(self.query_specs))

        # Pre-compile per-query template overrides — structural validation at
        # init time.  Templates are Tier 2 data: validated once at load, trusted
        # thereafter.  A TemplateSyntaxError here is a structural failure that
        # propagates up through LLMTransform.__init__ and stops the run at setup.
        query_templates: dict[str, PromptTemplate] = {}
        for spec in self.query_specs:
            if spec.template is not None:
                query_templates[spec.name] = self.template.with_template_override(spec.template)
        object.__setattr__(self, "_query_templates", query_templates)
        freeze_fields(self, "_query_templates")

    def execute(
        self,
        row: PipelineRow,
        ctx: TransformContext,
        *,
        provider: LLMProvider,
        tracer: LangfuseTracer,
    ) -> TransformResult:
        """Execute all queries, returning atomic success or failure."""
        state_id = ctx.state_id
        if state_id is None:
            raise RuntimeError("LLMTransform requires state_id")
        if ctx.token is None:
            raise RuntimeError("LLMTransform requires ctx.token")
        token_id = ctx.token.token_id
        shutdown_event = cast("threading.Event | None", getattr(ctx, "shutdown_event", None))

        if self.executor is not None:
            return self._execute_parallel(row, state_id, token_id, provider, tracer, shutdown_event)
        return self._execute_sequential(row, state_id, token_id, provider, tracer, shutdown_event)

    @dataclass(frozen=True, slots=True)
    class _QuerySuccess:
        """Partial output fields from a successful single-query execution.

        Tagged success type replacing bare ``dict`` in the union return of
        ``_execute_one_query``, so callers can exhaustively match on
        ``_QuerySuccess | TransformResult`` instead of ``dict | TransformResult``.
        """

        # ``Mapping`` (not ``dict``): ``freeze_fields`` replaces both
        # fields with ``MappingProxyType``; the annotation must describe
        # the read-only runtime type callers observe. Consumers needing
        # a writable dict must ``dict(result.fields)`` (see the
        # ``PipelineRow(dict(result.fields), ...)`` call below).
        fields: Mapping[str, Any]
        audit_metadata: Mapping[str, str | None]

        def __post_init__(self) -> None:
            freeze_fields(self, "fields", "audit_metadata")

    def _execute_one_query(
        self,
        query_idx: int,
        spec: QuerySpec,
        row: PipelineRow,
        state_id: str,
        token_id: str,
        provider: LLMProvider,
        tracer: LangfuseTracer,
        shutdown_event: threading.Event | None = None,
    ) -> _QuerySuccess | TransformResult:
        """Execute a single query within a multi-query row.

        Returns:
            _QuerySuccess: Partial output fields on success
            TransformResult: Error result on failure

        Raises:
            LLMClientError: Retryable errors propagate to caller for retry
                handling (PooledExecutor AIMD or sequential error capture).
        """
        if _shutdown_event_is_set(shutdown_event):
            return _shutdown_requested_result(query_name=spec.name, query_index=query_idx)

        # Build template context from named input_fields
        try:
            template_ctx = spec.build_template_context(row)
        except KeyError as e:
            return TransformResult.error(
                {
                    "reason": "template_context_failed",
                    "query_name": spec.name,
                    "query_index": query_idx,
                    "error": str(e),
                },
                retryable=False,
            )

        # Use pre-compiled per-query template (already structurally validated
        # in __post_init__), falling back to config-level template.
        if spec.template is not None:
            query_template = self._query_templates[spec.name]
        else:
            query_template = self.template

        # Render template — use contract=None because template_ctx is a
        # synthetic dict (keys are template variable names from input_fields,
        # not source column names). Passing the source row's contract would
        # wrap template_ctx in a PipelineRow that rejects these synthetic keys
        # in FIXED schema mode.
        try:
            rendered = query_template.render_with_metadata(
                template_ctx,
                contract=None,
            )
        except TemplateError as e:
            return TransformResult.error(
                {
                    "reason": "template_rendering_failed",
                    "query_name": spec.name,
                    "query_index": query_idx,
                    "error": str(e),
                    "template_hash": query_template.template_hash,
                },
                retryable=False,
            )

        # Build messages
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": rendered.prompt})

        if _shutdown_event_is_set(shutdown_event):
            return _shutdown_requested_result(query_name=spec.name, query_index=query_idx)

        # Execute query
        query_max_tokens = spec.max_tokens or self.max_tokens

        # Build response_format for structured output requests
        response_format: dict[str, Any] | None = None
        if spec.output_fields:
            if spec.response_format == ResponseFormat.STRUCTURED:
                properties = {f.suffix: f.to_json_schema() for f in spec.output_fields}
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{spec.name}_output",
                        "schema": {
                            "type": "object",
                            "properties": properties,
                            "required": list(properties),
                        },
                    },
                }
            else:
                response_format = {"type": "json_object"}

        start_time = time.monotonic()
        try:
            result = provider.execute_query(
                messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=query_max_tokens,
                state_id=state_id,
                token_id=token_id,
                response_format=response_format,
            )
        except ContextLengthError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            tracer.record_error(
                token_id=token_id,
                query_name=spec.name,
                prompt=rendered.prompt,
                error_message=str(e),
                model=self.model,
                latency_ms=latency_ms,
            )
            return TransformResult.error(
                {
                    "reason": "context_length_exceeded",
                    "failed_query_name": spec.name,
                    "failed_query_index": query_idx,
                    "error": str(e),
                },
                retryable=False,
            )
        except LLMClientError as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            tracer.record_error(
                token_id=token_id,
                query_name=spec.name,
                prompt=rendered.prompt,
                error_message=str(e),
                model=self.model,
                latency_ms=latency_ms,
            )
            if e.retryable:
                raise  # Pool catches with AIMD; sequential catches and returns error
            return TransformResult.error(
                {
                    "reason": "multi_query_failed",
                    "failed_query_name": spec.name,
                    "failed_query_index": query_idx,
                    "error": str(e),
                },
                retryable=False,
            )

        latency_ms = (time.monotonic() - start_time) * 1000

        # Fail closed on provider-signaled terminal finish reasons.
        finish_reason_error = _finish_reason_error(
            result.finish_reason,
            query_name=spec.name,
            query_index=query_idx,
        )
        if finish_reason_error is not None:
            tracer.record_error(
                token_id=token_id,
                query_name=spec.name,
                prompt=rendered.prompt,
                error_message=finish_reason_error.error_message,
                model=self.model,
                latency_ms=latency_ms,
            )
            return finish_reason_error.result

        # Strip fences and store content
        content = strip_markdown_fences(result.content)

        tracer.record_success(
            token_id=token_id,
            query_name=spec.name,
            prompt=rendered.prompt,
            response_content=content,
            model=self.model,
            usage=result.usage,
            latency_ms=latency_ms,
        )

        # Build partial output for this query
        partial: dict[str, Any] = {}

        # JSON parsing + field extraction when output_fields configured
        if spec.output_fields:
            # LLM response content is Tier 3 — parse and validate immediately
            try:
                parsed = json.loads(content, parse_constant=reject_nonfinite_constant)
            except (json.JSONDecodeError, ValueError) as e:
                return TransformResult.error(
                    {
                        "reason": "json_parse_failed",
                        "query_name": spec.name,
                        "query_index": query_idx,
                        "error": str(e),
                        "raw_response_preview": content[:500],
                    },
                    retryable=False,
                )
            if not isinstance(parsed, dict):
                return TransformResult.error(
                    {
                        "reason": "invalid_json_type",
                        "query_name": spec.name,
                        "query_index": query_idx,
                        "expected": "object",
                        "actual": type(parsed).__name__,
                    },
                    retryable=False,
                )
            # Extract typed fields into prefixed output columns.
            # Validate field presence — Tier 3 boundary: if the LLM omitted
            # a declared field, that's an error, not a silent None.
            for field in spec.output_fields:
                field_key = f"{spec.name}_{field.suffix}"
                if field.suffix not in parsed:
                    return TransformResult.error(
                        {
                            "reason": "missing_output_field",
                            "query_name": spec.name,
                            "query_index": query_idx,
                            "field": field.suffix,
                            "available_fields": list(parsed.keys()),
                        },
                        retryable=False,
                    )
                # Validate value type — Tier 3 boundary: LLM may return
                # wrong types in standard mode (no API schema enforcement)
                type_error = validate_field_value(parsed[field.suffix], field)
                if type_error is not None:
                    return TransformResult.error(
                        {
                            "reason": "field_type_mismatch",
                            "query_name": spec.name,
                            "query_index": query_idx,
                            "field": field.suffix,
                            "error": type_error,
                            "value": repr(parsed[field.suffix])[:200],
                        },
                        retryable=False,
                    )
                partial[field_key] = parsed[field.suffix]
            # Also store raw content for audit traceability
            partial[f"{spec.name}_{self.response_field}"] = content
        else:
            # Unstructured: store raw content only
            partial[f"{spec.name}_{self.response_field}"] = content

        populate_llm_operational_fields(
            partial,
            f"{spec.name}_{self.response_field}",
            usage=result.usage,
            model=result.model,
        )

        audit_metadata = build_llm_audit_metadata(
            f"{spec.name}_{self.response_field}",
            template_hash=rendered.template_hash,
            variables_hash=rendered.variables_hash,
            template_source=rendered.template_source,
            lookup_hash=rendered.lookup_hash,
            lookup_source=rendered.lookup_source,
            system_prompt_source=self.system_prompt_source,
        )
        # Record finish_reason so audit trail distinguishes None (absent) from
        # STOP (confirmed completion). See Bug elspeth-393d2459aa.
        audit_metadata[f"{spec.name}_finish_reason"] = _serialize_finish_reason(result.finish_reason)

        return self._QuerySuccess(fields=partial, audit_metadata=audit_metadata)

    def _execute_sequential(
        self,
        row: PipelineRow,
        state_id: str,
        token_id: str,
        provider: LLMProvider,
        tracer: LangfuseTracer,
        shutdown_event: threading.Event | None = None,
    ) -> TransformResult:
        """Execute queries sequentially (pool_size=1 fallback).

        Short-circuits on first error. Retryable LLMClientErrors are returned
        as error results instead of being re-raised, so the engine retry doesn't
        wastefully re-execute all queries from scratch.
        """
        accumulated_outputs: dict[str, Any] = {}
        accumulated_audit: dict[str, object] = {}

        for query_idx, spec in enumerate(self.query_specs):
            try:
                result = self._execute_one_query(
                    query_idx,
                    spec,
                    row,
                    state_id,
                    token_id,
                    provider,
                    tracer,
                    shutdown_event,
                )
            except LLMClientError as e:
                # Sequential mode: no AIMD retry — return retryable error result
                return TransformResult.error(
                    {
                        "reason": "multi_query_failed",
                        "failed_query_name": spec.name,
                        "failed_query_index": query_idx,
                        "error": str(e),
                        "discarded_successful_queries": query_idx,
                    },
                    retryable=e.retryable,
                )

            # Error from template/JSON/validation/non-retryable LLM
            if isinstance(result, TransformResult):
                # A TransformResult with status="error" and reason=None is a bug
                # in our code — mirrors the same guard in _execute_parallel.
                if result.reason is None:
                    raise FrameworkBugError(
                        f"Multi-query sequential execution produced TransformResult with "
                        f"status='error' but reason=None for query index {query_idx} "
                        f"(query_name={spec.name!r}). Every error path must set a reason dict."
                    )
                # Add discarded count for error reporting — copy first to avoid
                # mutating the original dict (which may be shared with audit records)
                augmented_reason = cast(TransformErrorReason, dict(result.reason))
                augmented_reason["discarded_successful_queries"] = query_idx
                return TransformResult.error(
                    augmented_reason,
                    retryable=result.retryable,
                    context_after=result.context_after,
                )

            accumulated_outputs.update(result.fields)
            accumulated_audit.update(result.audit_metadata)

        # All queries succeeded — build output row
        output = {**row.to_dict(), **accumulated_outputs}
        output_contract = propagate_contract(
            input_contract=row.contract,
            output_row=output,
            transform_adds_fields=True,
        )
        output_contract = self.align_output_contract(output_contract)

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "multi_query_enriched",
                "queries_completed": len(self.query_specs),
                "fields_added": list(accumulated_outputs.keys()),
                "metadata": {"model": self.model, **accumulated_audit},
            },
        )

    def _execute_parallel(
        self,
        row: PipelineRow,
        state_id: str,
        token_id: str,
        provider: LLMProvider,
        tracer: LangfuseTracer,
        shutdown_event: threading.Event | None = None,
    ) -> TransformResult:
        """Execute queries in parallel via PooledExecutor with AIMD retry.

        All queries run concurrently. The pool retries individual queries with
        AIMD backoff on retryable errors — only the failing query retries, not
        the entire row. Results are collected in submission order.
        """
        if self.executor is None:
            raise RuntimeError("_execute_parallel called without executor")

        # Side channel for audit metadata — _process_fn writes here, outer scope reads after pool.
        # Protected by lock: PooledExecutor runs _process_fn across ThreadPoolExecutor
        # workers, so concurrent dict writes require synchronization.
        # ``Mapping`` value type (not ``dict``): ``_QuerySuccess.audit_metadata``
        # is frozen via ``freeze_fields`` and surfaces as ``MappingProxyType``.
        # Declaring the slot as ``dict`` would make the assignment below a
        # silent type lie (see IM-2: ``dict[str, Any]`` after ``freeze_fields``).
        audit_metadata_by_index: dict[int, Mapping[str, str | None]] = {}
        audit_metadata_lock = threading.Lock()

        # Build RowContext for each query — the pool treats each as a "row"
        contexts = [
            RowContext(
                row={
                    "original_row": row,
                    "spec": spec,
                    "query_idx": i,
                    "provider": provider,
                    "tracer": tracer,
                    "token_id": token_id,
                    "state_id": state_id,
                },
                state_id=state_id,
                row_index=i,
            )
            for i, spec in enumerate(self.query_specs)
        ]

        def _process_fn(work: Mapping[str, Any], _work_state_id: str) -> TransformResult:
            """Pool process function — wraps _execute_one_query for pool interface."""
            result = self._execute_one_query(
                work["query_idx"],
                work["spec"],
                work["original_row"],
                work["state_id"],
                work["token_id"],
                work["provider"],
                work["tracer"],
                shutdown_event,
            )
            if isinstance(result, TransformResult):
                return result  # Error passthrough
            # Stash audit metadata in side channel before wrapping in TransformResult.
            # Lock required: multiple pool workers write concurrently.
            with audit_metadata_lock:
                audit_metadata_by_index[work["query_idx"]] = result.audit_metadata
            # Success: wrap partial fields in TransformResult for pool interface
            observed = SchemaContract(
                mode="OBSERVED",
                fields=tuple(
                    FieldContract(
                        normalized_name=k,
                        original_name=k,
                        python_type=type(v) if type(v) in (int, str, float, bool) else object,
                        required=False,
                        source="inferred",
                    )
                    for k, v in result.fields.items()
                ),
                locked=True,
            )
            # result.fields is a MappingProxyType (deep-frozen in
            # _QuerySuccess.__post_init__). PipelineRow enforces
            # ``type(data) is dict`` as a Tier 1 guard against silently coercing
            # audit/replay data — so materialize a fresh dict here. PipelineRow
            # re-deep-freezes internally, preserving immutability across the handoff.
            return TransformResult.success(
                self.align_output_row_contract(PipelineRow(dict(result.fields), observed)),
                success_reason={"action": "query_completed", "metadata": {"query_name": work["spec"].name}},
            )

        if shutdown_event is None:
            entries = self.executor.execute_batch(contexts=contexts, process_fn=_process_fn)
        else:
            entries = self.executor.execute_batch(contexts=contexts, process_fn=_process_fn, shutdown_event=shutdown_event)
        query_results = [entry.result for entry in entries]

        # Check for failures — atomic: any failure fails the row
        failed = [(i, r) for i, r in enumerate(query_results) if r.status == "error"]
        if failed:
            first_idx, first_result = failed[0]
            spec = self.query_specs[first_idx]
            # A TransformResult with status="error" and reason=None is a bug in
            # our code — every error path in _execute_one_query sets a reason dict.
            # Fabricating a reason here would hide the bug. Crash per offensive
            # programming policy.
            if first_result.reason is None:
                raise FrameworkBugError(
                    f"Multi-query parallel execution produced TransformResult with "
                    f"status='error' but reason=None for query index {first_idx} "
                    f"(query_name={spec.name!r}). Every error path must set a reason dict."
                )
            error_reason: TransformErrorReason = cast(
                TransformErrorReason,
                dict(first_result.reason),
            )
            error_reason["failed_query_name"] = spec.name
            error_reason["failed_query_index"] = first_idx
            error_reason["discarded_successful_queries"] = len(query_results) - len(failed)
            error_reason["failed_queries"] = [self.query_specs[i].name for i, _ in failed]
            error_reason["total_count"] = len(query_results)
            return TransformResult.error(error_reason, retryable=first_result.retryable)

        # Merge all successful partial outputs
        accumulated_outputs: dict[str, Any] = {}
        for result in query_results:
            if result.row is not None:
                accumulated_outputs.update(result.row.to_dict())

        # Merge audit metadata from side channel (written by _process_fn).
        # Validate that every successful query contributed its audit metadata —
        # a missing entry means incomplete provenance in the audit trail.
        success_indices = {i for i, r in enumerate(query_results) if r.status == "success"}
        missing_audit = success_indices - set(audit_metadata_by_index.keys())
        if missing_audit:
            raise FrameworkBugError(
                f"Multi-query parallel execution lost audit metadata for query indices "
                f"{sorted(missing_audit)}. Side-channel write in _process_fn did not "
                f"execute for these successful queries."
            )
        accumulated_audit: dict[str, object] = {}
        for idx in sorted(audit_metadata_by_index):
            accumulated_audit.update(audit_metadata_by_index[idx])

        output = {**row.to_dict(), **accumulated_outputs}
        output_contract = propagate_contract(
            input_contract=row.contract,
            output_row=output,
            transform_adds_fields=True,
        )
        output_contract = self.align_output_contract(output_contract)

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "multi_query_enriched",
                "queries_completed": len(self.query_specs),
                "fields_added": list(accumulated_outputs.keys()),
                "metadata": {"model": self.model, **accumulated_audit},
            },
        )


# ---------------------------------------------------------------------------
# LLMTransform — the unified plugin class
# ---------------------------------------------------------------------------


class LLMTransform(BaseTransform, BatchTransformMixin):
    """Unified LLM transform with provider dispatch and strategy selection.

    Registered as plugin name="llm". Uses BatchTransformMixin for concurrent
    row processing with FIFO output ordering and backpressure.

    Provider dispatch:
        "azure"      → AzureOpenAIConfig + AzureLLMProvider
        "openrouter" → OpenRouterConfig  + OpenRouterLLMProvider

    Strategy selection:
        queries is not None → MultiQueryStrategy
        queries is None     → SingleQueryStrategy
    """

    name = "llm"
    requires_runtime_preflight = True
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:3024fee57fccc324"
    determinism: Determinism = Determinism.NON_DETERMINISTIC
    config_model = LLMConfig  # Base; get_config_model dispatches to provider-specific
    passes_through_input = True
    _provider: LLMProvider | None

    @classmethod
    def get_config_model(cls, config: dict[str, Any] | None = None) -> type[LLMConfig]:
        """Dispatch to provider-specific config class based on config["provider"]."""
        provider = config.get("provider") if config is not None else None
        if provider is not None and provider in _PROVIDERS:
            config_cls, _ = _PROVIDERS[provider]
            return config_cls
        elif provider is not None:
            raise ValueError(f"Unknown LLM provider '{provider}'. Valid providers: {sorted(_PROVIDERS)}")
        # provider missing — return base LLMConfig so Pydantic catches it with Literal validation
        return LLMConfig

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Publish the full LLM config contract as a Pydantic discriminated union.

        The runtime dispatches ``get_config_model(config)`` on ``config["provider"]``
        to select the provider-specific config class registered in
        :data:`_PROVIDERS`. The catalog needs to advertise the complete
        contract at schema-discovery time — before any provider has been
        chosen — so this returns ``oneOf`` + a ``provider`` discriminator
        over every variant in ``_PROVIDERS``. The variant list is derived
        from the same registry that drives runtime provider instantiation;
        the two cannot drift.

        Requires ``len(_PROVIDERS) >= 2`` because a discriminated union with
        one branch degenerates — Pydantic's ``TypeAdapter`` would emit a
        non-``oneOf`` schema and downstream consumers (catalog flattener,
        frontend PluginCard) rely on the ``oneOf`` + ``$defs`` shape to route
        rendering. The guard here surfaces the contract explicitly instead of
        letting the shape silently change under a single-provider build.
        """
        variants = tuple(cfg for cfg, _ in _PROVIDERS.values())
        if len(variants) < 2:
            raise RuntimeError(
                "LLMTransform.get_config_schema requires at least two providers "
                f"registered in _PROVIDERS; found {len(variants)}. Discriminated "
                "union shape requires multiple branches."
            )
        # Fold with ``|`` to a single types.UnionType — the form Pydantic's
        # Annotated[..., Field(discriminator=...)] requires. typing.Union[*variants]
        # and tuple-of-types do not produce the same schema shape.
        union_type = reduce(or_, variants[1:], variants[0])
        adapter: TypeAdapter[Any] = TypeAdapter(Annotated[union_type, PydanticField(discriminator="provider")])
        schema: dict[str, Any] = adapter.json_schema(ref_template="#/$defs/{model}")
        # Pydantic emits ``provider`` as ``{const, default}`` on each variant and,
        # because the field has a default, omits it from the variant's ``required``
        # set. That disagrees with the runtime contract: ``LLMTransform.__init__``
        # raises ``ValueError`` when config lacks ``provider`` because the registry
        # cannot dispatch without it. A JSON-Schema preflight consumer (MCP composer,
        # PluginCard form generation) must see the same requirement the runtime
        # enforces — otherwise it accepts configs the pipeline will reject.
        # Republish ``provider`` as required on every variant defined by ``_PROVIDERS``.
        # Targeting variants by class name (not by "has a provider property") keeps
        # the injection scoped to the discriminated union; unrelated nested ``$defs``
        # (e.g. ``SchemaConfig``) are untouched.
        provider_variant_names = {cfg.__name__ for cfg, _ in _PROVIDERS.values()}
        for variant_name in provider_variant_names:
            variant_schema = schema["$defs"][variant_name]
            # Each provider variant inherits non-empty required fields from
            # LLMConfig (``schema``, ``template``, ...) so Pydantic always
            # emits a ``required`` key on the variant $defs. Access it directly
            # — a missing key would signal a schema-emission regression worth
            # crashing on, not silently papering over with setdefault().
            required = variant_schema["required"]
            if "provider" not in required:
                required.append("provider")
        return schema

    @classmethod
    def discriminated_variants(cls) -> tuple[str, dict[str, type[BaseModel]]]:
        """Expose provider variants to composer knob-schema lowering."""
        return ("provider", {provider: config_cls for provider, (config_cls, _) in _PROVIDERS.items()})

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal no-network config for the ADR-009 forward invariant.

        Uses ``openai/gpt-4o`` because it's a stable presence in
        ``litellm.model_list``'s OpenRouter slice — the probe never
        actually makes a network call (preflight mode defers all
        external client setup), but the value-source compliance walker
        runs at construction time, so the model identifier MUST satisfy
        the OpenRouter catalog declaration. A model that drops in/out
        across litellm versions (e.g. ``gpt-4o-mini``) would make probes
        flaky against the catalog snapshot.
        """
        return {
            "provider": "openrouter",
            "api_key": "probe-key",
            "model": "openai/gpt-4o",
            "prompt_template": "{{ row.llm_probe_text }}",
            "schema": {"mode": "observed"},
            "required_input_fields": [],
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Inject a deterministic prompt field for invariant probing."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="llm_probe_text",
                value="probe request",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        """Exercise the real LLM row path with a lifecycle-safe fake provider."""
        if len(probe_rows) != 1:
            raise FrameworkBugError(
                f"{self.__class__.__name__}.execute_forward_invariant_probe() "
                f"received {len(probe_rows)} rows; LLM invariant probes require exactly 1 row."
            )

        class _InvariantProvider:
            def execute_query(
                self,
                messages: list[dict[str, str]],
                *,
                model: str,
                temperature: float,
                max_tokens: int | None,
                state_id: str,
                token_id: str,
                response_format: object | None = None,
            ) -> LLMQueryResult:
                del messages, model, temperature, max_tokens, state_id, token_id, response_format
                return LLMQueryResult(
                    content="probe response",
                    usage=TokenUsage.known(1, 1),
                    model="probe-model",
                    finish_reason=FinishReason.STOP,
                )

            def runtime_preflight(self, *, operation_id: str, model: str) -> None:
                del operation_id, model

            def close(self) -> None:
                return None

        original_provider = self._provider
        probe_provider = _InvariantProvider()
        try:
            self._provider = probe_provider
            return self._process_row(probe_rows[0], ctx)
        finally:
            self._provider = original_provider
            probe_provider.close()

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        # Provider dispatch from single registry.
        # config is user YAML (Tier 3 boundary) — distinguish missing key from unknown value.
        provider_name = config.get("provider")
        if provider_name is None:
            raise ValueError(f"LLM config missing required 'provider' key. Valid providers: {sorted(_PROVIDERS)}")
        if provider_name not in _PROVIDERS:
            raise ValueError(f"Unknown LLM provider '{provider_name}'. Valid providers: {sorted(_PROVIDERS)}")
        config_cls, _ = _PROVIDERS[provider_name]

        # Parse config with provider-specific model.
        # config_cls is AzureOpenAIConfig or OpenRouterConfig at runtime;
        # from_dict() returns Self on the subclass, but mypy sees type[LLMConfig].
        self._config = cast(
            "AzureOpenAIConfig | OpenRouterConfig",
            config_cls.from_dict(config, plugin_name=self.name),
        )
        self._initialize_declared_input_fields(self._config)

        # Store common LLM settings.
        # AzureOpenAIConfig._set_model_from_deployment ensures model is populated;
        # OpenRouterConfig requires model. So self._config.model is always non-empty.
        self._model = self._config.model
        self._template = PromptTemplate(
            self._config.prompt_template,
            template_source=self._config.prompt_template_source,
            lookup_data=self._config.lookup,
            lookup_source=self._config.lookup_source,
        )
        self._system_prompt = self._config.system_prompt
        self._system_prompt_source = self._config.system_prompt_source
        self._temperature = self._config.temperature
        self._max_tokens = self._config.max_tokens
        self._response_field = self._config.response_field
        self._max_capacity_retry_seconds = self._config.max_capacity_retry_seconds
        self._pool_size = self._config.pool_size
        # Phase 5b Task 9 — cross-DB hash anchor. ``None`` when this LLM
        # transform was not staged via an interpretation event; populated
        # by the session service's ``resolve_interpretation_event`` writer
        # when the transform's prompt template carried a
        # ``{{interpretation:<term>}}`` placeholder the user resolved.
        # Forwarded to every audited LLM call so the Landscape
        # ``calls.resolved_prompt_template_hash`` column populates.
        self._resolved_prompt_template_hash = self._config.resolved_prompt_template_hash

        # Schema (input — same for both single and multi-query)
        schema_config = self._config.schema_config
        self.input_schema = create_schema_from_config(
            schema_config,
            f"{self.name}Schema",
            allow_coercion=False,
        )
        # output_schema and _output_schema_config are set in the strategy
        # dispatch below — multi-query uses prefixed fields, single-query
        # uses unprefixed fields.

        # Tracer — factory returns frozen ActiveLangfuseTracer or NoOpLangfuseTracer.
        # tracing lives on provider-specific configs (AzureOpenAIConfig, OpenRouterConfig),
        # both of which define tracing: dict[str, Any] | None.
        tracing_config = parse_tracing_config(self._config.tracing) if self._config.tracing else None
        self._tracing_config: TracingConfig | None = tracing_config

        # Validate provider/tracing compatibility — azure_ai tracing auto-instruments
        # the OpenAI SDK, which only the Azure provider uses. Fail loud, not silent.
        if isinstance(tracing_config, AzureAITracingConfig) and not isinstance(self._config, AzureOpenAIConfig):
            raise ValueError(
                "azure_ai tracing requires the azure provider. "
                "Azure Monitor auto-instruments the OpenAI SDK, which is only used by provider='azure'. "
                f"Current provider: '{self._config.provider}'"
            )

        self._tracer = create_langfuse_tracer(
            transform_name=self.name,
            tracing_config=tracing_config,
        )

        # Query-level executor for parallel multi-query execution with AIMD backoff.
        # Created here (not in on_start) because it doesn't depend on recorder/telemetry.
        pool_config = self._config.pool_config
        self._query_executor: PooledExecutor | None = PooledExecutor(pool_config) if pool_config is not None else None

        # Strategy dispatch: queries is not None → multi-query
        if self._config.queries is not None:
            query_specs = resolve_queries(self._config.queries)
            self._strategy: SingleQueryStrategy | MultiQueryStrategy = MultiQueryStrategy(
                query_specs=query_specs,
                template=self._template,
                system_prompt=self._system_prompt,
                system_prompt_source=self._system_prompt_source,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_field=self._response_field,
                align_output_contract=self._align_output_contract,
                align_output_row_contract=self._align_output_row_contract,
                executor=self._query_executor,
            )

            # Multi-query emits prefixed fields — compute guaranteed field sets
            # (audit fields now travel via success_reason, not the row)
            prefixed_guaranteed: set[str] = set()
            for spec in query_specs:
                prefix = f"{spec.name}_{self._response_field}"
                prefixed_guaranteed.add(prefix)
                prefixed_guaranteed.update(get_llm_guaranteed_fields(prefix))
                if spec.output_fields:
                    for field in spec.output_fields:
                        prefixed_guaranteed.add(f"{spec.name}_{field.suffix}")
            self.declared_output_fields = frozenset(prefixed_guaranteed)

            # Output schema config with prefixed fields for DAG contract propagation.
            # INVARIANT: guaranteed_fields must be a superset of declared_output_fields.
            # See: docs/superpowers/specs/2026-03-20-output-schema-contract-enforcement-design.md
            self._output_schema_config = _build_llm_output_schema_config(schema_config, prefixed_guaranteed)

            # Pydantic output schema with prefixed LLM fields
            # Build extracted_fields mapping: query_name → (field_name, schema_type) tuples
            extracted: dict[str, tuple[tuple[str, _FieldType], ...]] = {}
            for spec in query_specs:
                if spec.output_fields:
                    extracted[spec.name] = tuple(
                        (f"{spec.name}_{f.suffix}", _OUTPUT_FIELD_TYPE_TO_SCHEMA[f.type.value]) for f in spec.output_fields
                    )
            self.output_schema = _build_multi_query_output_schema(
                base_schema_config=schema_config,
                response_field=self._response_field,
                query_names=tuple(spec.name for spec in query_specs),
                schema_name=f"{self.name}OutputSchema",
                extracted_fields=extracted if extracted else None,
            )
        else:
            self._strategy = SingleQueryStrategy(
                template=self._template,
                system_prompt=self._system_prompt,
                system_prompt_source=self._system_prompt_source,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_field=self._response_field,
                align_output_contract=self._align_output_contract,
            )

            # Single-query emits unprefixed fields (operational only — audit goes to success_reason)
            guaranteed = get_llm_guaranteed_fields(self._response_field)
            self.declared_output_fields = frozenset(guaranteed)

            # Output schema config with LLM output fields for DAG contract propagation.
            # INVARIANT: guaranteed_fields must be a superset of declared_output_fields.
            # See: docs/superpowers/specs/2026-03-20-output-schema-contract-enforcement-design.md
            self._output_schema_config = _build_llm_output_schema_config(schema_config, guaranteed)

            # Pydantic output schema with unprefixed LLM fields
            self.output_schema = _build_augmented_output_schema(
                base_schema_config=schema_config,
                response_field=self._response_field,
                schema_name=f"{self.name}OutputSchema",
            )

        # Provider instance — deferred to on_start() when recorder/telemetry available
        self._provider: LLMProvider | None = None

        # Recorder, telemetry, rate limit (set in on_start)
        self._recorder: PluginAuditWriter | None = None
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = _warn_telemetry_before_start
        self._limiter: Any = None
        self._shutdown_event: threading.Event | None = None

        # Batch processing state
        self._batch_initialized = False

    @property
    def provider_config(self) -> AzureOpenAIConfig | OpenRouterConfig:
        """Read-only accessor for the typed provider config.

        Exposes the post-validation ``LLMConfig`` subclass so cross-cutting
        validators (notably the value-source compliance walker) can inspect
        ``VALUE_SOURCES`` declarations and field values without reaching for
        ``self._config`` private attribute access.

        Named ``provider_config`` (not ``config``) to avoid colliding with
        :class:`BaseTransform.config` (the raw ``dict[str, Any]`` configuration
        passed to ``__init__``) — distinct meanings, distinct names.
        """
        return self._config

    def output_semantics(self) -> OutputSemanticDeclaration:
        """Declare that raw LLM response fields are strings.

        Single-query mode emits ``response_field`` directly. Multi-query mode
        emits one raw response field per query using ``<query>_<response_field>``.
        Structured multi-query extracted fields have their own schema types, but
        this semantic contract only claims the raw response-content fields whose
        runtime value is mechanically known here.
        """
        from elspeth.contracts.plugin_semantics import (
            ContentKind,
            FieldSemanticFacts,
            OutputSemanticDeclaration,
            SemanticValueType,
            TextFraming,
        )

        if isinstance(self._strategy, MultiQueryStrategy):
            return OutputSemanticDeclaration(
                fields=tuple(
                    FieldSemanticFacts(
                        field_name=f"{spec.name}_{self._response_field}",
                        content_kind=ContentKind.UNKNOWN,
                        text_framing=TextFraming.UNKNOWN,
                        value_type=SemanticValueType.STR,
                        fact_code="llm.response_field.string",
                        configured_by=("queries", "response_field"),
                    )
                    for spec in self._strategy.query_specs
                ),
            )

        return OutputSemanticDeclaration(
            fields=(
                FieldSemanticFacts(
                    field_name=self._response_field,
                    content_kind=ContentKind.UNKNOWN,
                    text_framing=TextFraming.UNKNOWN,
                    value_type=SemanticValueType.STR,
                    fact_code="llm.response_field.string",
                    configured_by=("response_field", "queries"),
                ),
            ),
        )

    def connect_output(self, output: OutputPort, max_pending: int = 30) -> None:
        """Connect output port and initialize batch processing."""
        if self._batch_initialized:
            raise RuntimeError("connect_output() already called")

        self.init_batch_processing(
            max_pending=max_pending,
            output=output,
            name=self.name,
            max_workers=max_pending,
            batch_wait_timeout=float(self._max_capacity_retry_seconds),
        )
        self._batch_initialized = True

    def on_start(self, ctx: LifecycleContext) -> None:
        """Capture recorder/telemetry and create provider instance."""
        super().on_start(ctx)
        self._recorder = ctx.landscape
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit
        self._shutdown_event = ctx.shutdown_event
        limiter_name = "azure_openai" if isinstance(self._config, AzureOpenAIConfig) else "openrouter"
        self._limiter = ctx.rate_limit_registry.get_limiter(limiter_name) if ctx.rate_limit_registry is not None else None

        # Create provider now that recorder/telemetry are available
        self._provider = self._create_provider()

        # Initialize Azure AI tracing (process-level OpenTelemetry auto-instrumentation).
        # Must happen after provider creation — the OpenAI SDK must be available.
        if isinstance(self._tracing_config, AzureAITracingConfig):
            _configure_azure_monitor(self._tracing_config)
            logger.info(
                "Azure AI tracing initialized",
                provider="azure_ai",
                content_recording=self._tracing_config.enable_content_recording,
            )

    def runtime_preflight(self, ctx: LifecycleContext) -> None:
        """Fail fast if the configured LLM provider/model cannot be reached."""
        if self._provider is None:
            raise FrameworkBugError("LLMTransform runtime_preflight called before provider initialization")
        if ctx.operation_id is None:
            raise FrameworkBugError("LLMTransform runtime_preflight requires an operation audit parent")

        try:
            self._provider.runtime_preflight(operation_id=ctx.operation_id, model=self._model)
        except LLMClientError as exc:
            raise RuntimePreflightFailedError(
                plugin_name=self.name,
                provider=self._config.provider,
                cause=exc,
            ) from exc

    def _create_provider(self) -> LLMProvider:
        """Instantiate the provider with all required dependencies.

        Uses isinstance narrowing on self._config to safely access
        provider-specific attributes (endpoint, deployment_name, base_url, etc.).
        """
        if self._recorder is None:
            raise RuntimeError("_recorder not initialized — _create_provider called before on_start()")

        if isinstance(self._config, AzureOpenAIConfig):
            return AzureLLMProvider(
                endpoint=self._config.endpoint,
                api_key=self._config.api_key,
                api_version=self._config.api_version,
                deployment_name=self._config.deployment_name,
                recorder=self._recorder,
                run_id=self._run_id,
                telemetry_emit=self._telemetry_emit,
                limiter=self._limiter,
                resolved_prompt_template_hash=self._resolved_prompt_template_hash,
            )
        elif isinstance(self._config, OpenRouterConfig):
            return OpenRouterLLMProvider(
                api_key=self._config.api_key,
                base_url=self._config.base_url,
                timeout_seconds=self._config.timeout_seconds,
                recorder=self._recorder,
                run_id=self._run_id,
                telemetry_emit=self._telemetry_emit,
                limiter=self._limiter,
                resolved_prompt_template_hash=self._resolved_prompt_template_hash,
            )
        else:
            raise RuntimeError(f"Unknown config type: {type(self._config).__name__}")

    def accept(self, row: PipelineRow, ctx: TransformContext) -> None:
        """Accept a row for processing (pipeline entry point)."""
        if not self._batch_initialized:
            raise RuntimeError("connect_output() must be called before accept()")
        self.accept_row(row, ctx, self._process_row)

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Not supported — use accept() for row-level pipelining."""
        raise NotImplementedError(
            f"{self.__class__.__name__} uses row-level pipelining. Use accept() instead of process(). See class docstring for usage."
        )

    def _process_row(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Process a single row via the selected strategy.

        Called by worker threads from BatchTransformMixin. Delegates to
        SingleQueryStrategy or MultiQueryStrategy.
        """
        if self._provider is None:
            raise RuntimeError("Provider not initialized — _process_row called before on_start()")
        self._shutdown_event = cast("threading.Event | None", getattr(ctx, "shutdown_event", None))

        return self._strategy.execute(
            row,
            ctx,
            provider=self._provider,
            tracer=self._tracer,
        )

    def close(self) -> None:
        """Release resources."""
        self._tracer.flush()
        shutdown_requested = _shutdown_event_is_set(self._shutdown_event)

        if shutdown_requested and self._provider is not None:
            self._provider.close()
            self._provider = None

        if self._query_executor is not None:
            self._query_executor.shutdown(wait=not shutdown_requested)

        if self._batch_initialized:
            self.shutdown_batch_processing(wait_for_workers=not shutdown_requested)

        if self._provider is not None:
            self._provider.close()
            self._provider = None

        self._recorder = None
        self._shutdown_event = None

    @classmethod
    def get_agent_assistance(cls, *, issue_code: str | None = None) -> PluginAssistance | None:
        if issue_code is None:
            return PluginAssistance(
                plugin_name="llm",
                issue_code=None,
                summary="Call an LLM provider (Azure OpenAI or OpenRouter) on each row and write the response into a field. Tracks model identity, token usage, and finish reason in the audit trail.",
                composer_hints=(
                    "Call list_models before pinning 'model:' — deployments don't all ship the same providers.",
                    "If you create a prompt template from the user's goal, data, or prose instead of copying it verbatim, stage an llm_prompt_template review for the authored prompt text.",
                    "When you author LLM judgment semantics — a scoring scale, rubric, category meaning, threshold, signal weighting, cutoff, comparison set, or subjective criterion definition — stage a vague_term review on the LLM node before set_pipeline.",
                    "Measurable adjectives are not exempt: if the user gives the metric/cutoff, use it; if you choose a cutoff such as 'over 6 ft' or a ranking rule such as 'top quartile', review that authored threshold semantics.",
                    "Prompt-template review is not enough for authored judgment semantics: put both interpretation_requirements in the LLM node options before set_pipeline — one vague_term for the rubric/definition/threshold/category semantics and one llm_prompt_template for the raw prompt. Use a stable user_term preserving the user's criterion phrase, not the whole task phrase; for an adjective embedded in prose, use the adjective or noun phrase that names the criterion. llm_draft must equal the semantics you wrote; only an llm_prompt_template review is incomplete.",
                    "When repairing or upserting an LLM node, repeat the review preflight; carry forward existing pending LLM interpretation requirements and add missing vague_term or prompt shield requirements before stopping.",
                    "Do not stop by saying the rubric is part of the reviewed prompt. If you wrote the scale, label meanings, thresholds, cutoff, comparison rule, tie break, or signals, add the separate vague_term requirement and request its review.",
                    "If the prompt asks the model to return a score, rating, rank, class, or pass/fail result, that output shape is authored judgement semantics when you chose the scale, class meaning, threshold, or ranking rule.",
                    "Objective extraction such as identifying primary colours used does not by itself need a vague_term review unless you add subjective scoring, ranking, thresholds, or category semantics.",
                    "For how <adjective> phrasing, use the adjective itself as user_term unless the user supplied a more specific criterion phrase; do not use the whole how-phrase.",
                    "Token-usage and model-ID fields are appended automatically as <response_field>_usage / _model — don't hand-add them.",
                    "If downstream cleanup, sink, mapper, or transform needs the LLM response, guarantee the response_field by name in the LLM node schema. If downstream also needs source or scrape fields that pass through the LLM, also guarantee pass-through fields such as URL or identifier fields.",
                    "Single-query LLM output is written to response_field. Prompt-requested JSON keys are not separate pipeline fields unless another transform parses them; preserve response_field through cleanup instead of invented prompt-internal keys.",
                    "The LLM transform preserves upstream row fields while adding response_field; it does not remove raw scrape fields. If a web_scrape-to-LLM workflow must save results without raw HTML or fingerprints, put a field_mapper cleanup node between the LLM and the sink.",
                    "If internet content, public web content, search results, crawled pages, or other untrusted remote text will flow into this LLM, prompt-injection shielding is important: surface this to the user as a strong recommendation. Recommend azure_prompt_shield or the deployment's equivalent before the LLM; continuing without it is allowed, and recommendation is not permission to add a node automatically — do not insert it automatically unless requested, policy-required, or explicitly high-risk.",
                    "If the workflow proceeds without an authorized prompt shield, surface a pipeline_decision warning on the LLM node with user_term prompt_injection_shield_recommendation and a draft recommending azure_prompt_shield while stating that internet-controlled text will otherwise flow directly to the LLM.",
                    "LLM-node reviews stack: an authored web-content scoring prompt can need llm_prompt_template, vague_term, and a prompt_injection_shield_recommendation warning on the same LLM node.",
                    "Interpretation reviews are not transform stages. Do not create passthrough, review, recommendation, or placeholder nodes for LLM reviews; put the review objects in this LLM node's interpretation_requirements list.",
                    "For prompt-injection shielding recommendations, do not add passthrough, placeholder, no-op, or renamed utility nodes to imply protection; recommendation prose is not a graph step.",
                    "This is prompt-injection defense; do not substitute azure_content_safety. Use azure_content_safety only for harmful-content moderation or safety classification.",
                    "max_concurrency and per_minute_rate_limit interact — neither bounds the other; set both when the provider has hard rate caps.",
                ),
            )
        return None

    @classmethod
    def get_post_call_hints(
        cls,
        *,
        tool_name: str,
        config_snapshot: Mapping[str, object],
    ) -> tuple[str, ...]:
        hints: list[str] = []
        # Manual _usage / _model field declared by hand → tell them it's automatic.
        if "response_field" in config_snapshot and "output_schema" in config_snapshot:
            response_field = config_snapshot["response_field"]
            output_schema = config_snapshot["output_schema"]
            if isinstance(response_field, str) and isinstance(output_schema, Mapping) and "fields" in output_schema:
                fields = output_schema["fields"]
                if isinstance(fields, Sequence):
                    manual_appendix = {f"{response_field}_usage", f"{response_field}_model"}
                    declared = {field.split(":", 1)[0].strip() if isinstance(field, str) else "" for field in fields}
                    if manual_appendix & declared:
                        hints.append(
                            f"You declared {sorted(manual_appendix & declared)!r} in the schema, but token-usage and model-ID fields are appended automatically. Remove them from output_schema.fields."
                        )
        return tuple(hints)


# Register opt-in for value-source compliance: the typed Pydantic config
# (OpenRouterConfig | AzureOpenAIConfig) is exposed via the public
# ``provider_config`` property. The walker (engine/orchestrator/preflight.py)
# uses this registration to find the typed config without resorting to
# duck-typing or getattr.
register_value_source_plugin(LLMTransform, config_attr="provider_config")
