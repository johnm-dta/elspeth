"""RAG retrieval transform — enriches rows with context from vector/keyword search.

Lifecycle:
    __init__: Parse config, build QueryBuilder, initialize accumulators.
    on_start: Construct provider via PROVIDERS registry factory.
    process: Build query -> search -> format -> attach to row.
    on_complete: Emit telemetry with run statistics.
    close: Release provider and query builder resources.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import structlog

from elspeth.contracts import Determinism, TransformResult, propagate_contract
from elspeth.contracts.errors import FrameworkBugError, RetrievalNotReadyError, TransformErrorReason
from elspeth.contracts.freeze import deep_thaw
from elspeth.contracts.schema_contract import PipelineRow
from elspeth.plugins.infrastructure.base import BaseTransform
from elspeth.plugins.infrastructure.clients.retrieval.base import RetrievalError
from elspeth.plugins.infrastructure.clients.retrieval.types import RetrievalChunk
from elspeth.plugins.transforms.rag.config import PROVIDERS, RAGRetrievalConfig
from elspeth.plugins.transforms.rag.formatter import format_context
from elspeth.plugins.transforms.rag.query import QueryBuilder

if TYPE_CHECKING:
    from elspeth.contracts.contexts import LifecycleContext, TransformContext
    from elspeth.plugins.infrastructure.clients.retrieval.base import RetrievalProvider

logger = structlog.get_logger(__name__)


def _warn_telemetry_before_start(event: Any) -> None:
    """Default telemetry callback before on_start() — warns instead of silently dropping."""
    logger.warning(
        "telemetry_emit called before on_start() — event dropped",
        event_type=type(event).__name__,
    )


class RAGRetrievalTransform(BaseTransform):
    """Enriches rows with retrieval-augmented context from search providers.

    Registered as plugin name="rag_retrieval". Uses synchronous process()
    since retrieval calls are I/O-bound but single-query-per-row.

    Output fields (prefixed with output_prefix):
        {prefix}__rag_context: Formatted text from retrieved chunks.
        {prefix}__rag_score: Best relevance score (float, 0.0-1.0).
        {prefix}__rag_count: Number of chunks retrieved (int).
        {prefix}__rag_sources: JSON envelope with source provenance.
    """

    name = "rag_retrieval"
    plugin_version = "1.0.0"
    source_file_hash: str | None = "sha256:9026ebd1220106b1"
    determinism: Determinism = Determinism.EXTERNAL_CALL
    config_model = RAGRetrievalConfig
    passes_through_input = True
    _provider: RetrievalProvider | None

    @classmethod
    def probe_config(cls) -> dict[str, Any]:
        """Minimal no-network config for the ADR-009 forward invariant."""
        return {
            "output_prefix": "policy",
            "query_field": "rag_probe_query",
            "provider": "azure_search",
            "provider_config": {
                "endpoint": "https://invariant.example.search.windows.net",
                "index": "invariant-probe",
                "api_key": "probe-key",
            },
            "schema": {"mode": "observed"},
        }

    def forward_invariant_probe_rows(self, probe: PipelineRow) -> list[PipelineRow]:
        """Inject a deterministic retrieval query for invariant probing."""
        return [
            self._augment_invariant_probe_row(
                probe,
                field_name="rag_probe_query",
                value="What is the policy?",
            )
        ]

    def execute_forward_invariant_probe(
        self,
        probe_rows: list[PipelineRow],
        ctx: Any,
    ) -> TransformResult:
        """Exercise the real retrieval path with a provider-agnostic local double."""
        if len(probe_rows) != 1:
            raise FrameworkBugError(
                f"{self.__class__.__name__}.execute_forward_invariant_probe() "
                f"received {len(probe_rows)} rows; RAG invariant probes require exactly 1 row."
            )

        class _InvariantProvider:
            def __init__(self) -> None:
                self.last_skipped_count = 0
                self.last_skipped_reasons: list[dict[str, Any]] = []

            def search(
                self,
                query: str,
                top_k: int,
                min_score: float,
                *,
                state_id: str,
                token_id: str | None,
            ) -> list[RetrievalChunk]:
                del query, top_k, min_score, state_id, token_id
                return [
                    RetrievalChunk(
                        content="Probe context",
                        score=0.95,
                        source_id="probe-doc",
                        metadata={"kind": "invariant"},
                    )
                ]

            def close(self) -> None:
                return None

        original_provider = self._provider
        original_on_start_called = self._on_start_called
        probe_provider = _InvariantProvider()
        try:
            self.__dict__["_provider"] = probe_provider
            self._on_start_called = True
            return super().execute_forward_invariant_probe(probe_rows, ctx)
        finally:
            self._on_start_called = original_on_start_called
            self.__dict__["_provider"] = original_provider
            probe_provider.close()

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self._rag_config = RAGRetrievalConfig.from_dict(config, plugin_name=self.name)
        self._initialize_declared_input_fields(self._rag_config)
        prefix = self._rag_config.output_prefix

        # Output field names
        self._field_context = f"{prefix}__rag_context"
        self._field_score = f"{prefix}__rag_score"
        self._field_count = f"{prefix}__rag_count"
        self._field_sources = f"{prefix}__rag_sources"

        self.declared_output_fields = frozenset(
            [
                self._field_context,
                self._field_score,
                self._field_count,
                self._field_sources,
            ]
        )

        # Schemas — RAG adds fields, so output uses observed mode
        self.input_schema, self.output_schema = self._create_schemas(
            self._rag_config.schema_config,
            self.name,
            adds_fields=True,
        )

        # Output schema config for DAG contract propagation.
        self._output_schema_config = self._build_output_schema_config(self._rag_config.schema_config)

        # Query builder
        self._query_builder = QueryBuilder(
            self._rag_config.query_field,
            query_template=self._rag_config.query_template,
            query_pattern=self._rag_config.query_pattern,
        )

        # Welford online accumulators for telemetry
        self._total_queries = 0
        self._quarantine_count = 0
        self._total_chunks = 0
        self._score_count = 0
        self._score_mean = 0.0
        self._score_m2 = 0.0

        # Provider — deferred to on_start()
        self._provider: RetrievalProvider | None = None

        # Lifecycle dependencies — set in on_start()
        self._run_id: str = ""
        self._telemetry_emit: Callable[[Any], None] = _warn_telemetry_before_start

    def on_start(self, ctx: LifecycleContext) -> None:
        """Capture lifecycle context and construct the search provider."""
        super().on_start(ctx)
        self._run_id = ctx.run_id
        self._telemetry_emit = ctx.telemetry_emit

        # Construct provider from registry
        provider_name = self._rag_config.provider
        config_cls, factory = PROVIDERS[provider_name]
        provider_config = config_cls(**self._rag_config.provider_config)
        collection_name = self._configured_collection_name(provider_config)

        try:
            self._provider = factory(
                provider_config,
                execution=ctx.landscape,
                run_id=ctx.run_id,
                telemetry_emit=ctx.telemetry_emit,
                limiter=(ctx.rate_limit_registry.get_limiter(provider_name) if ctx.rate_limit_registry is not None else None),
            )

            # Readiness check — refuse to start against empty/missing collection.
            # Two distinct failure modes: unreachable (infra problem) and empty
            # (operator error). Both crash startup, but the message distinguishes them.
            readiness = self._provider.check_readiness()
        except RetrievalError as exc:
            self._record_readiness_check(
                ctx,
                collection=collection_name,
                reachable=False,
                count=None,
                message=str(exc),
            )
            raise RetrievalNotReadyError(collection=collection_name, reason=str(exc)) from exc

        # Record first — the readiness result is an auditable fact regardless of outcome.
        # "If it's not recorded, it didn't happen" — an auditor can query
        # what the collection state was when this pipeline started, including failures.
        self._record_readiness_check(
            ctx,
            collection=readiness.collection,
            reachable=readiness.reachable,
            count=readiness.count,
            message=readiness.message,
        )

        # Then guard on the result
        if not readiness.reachable or readiness.count is None or readiness.count <= 0:
            raise RetrievalNotReadyError(
                collection=readiness.collection,
                reason=readiness.message,
            )

    def process(self, row: PipelineRow, ctx: TransformContext) -> TransformResult:
        """Process a single row: build query, search, format, attach."""
        if not self._on_start_called:
            raise RuntimeError(
                f"{self.__class__.__name__}.process() called before on_start(). "
                f"The orchestrator must call on_start() before processing rows."
            )
        if self._provider is None:
            raise RuntimeError(
                f"{self.__class__.__name__} provider not initialized. on_start() must construct the provider before process() is called."
            )
        if ctx.state_id is None:
            raise RuntimeError(f"{self.__class__.__name__} requires state_id on TransformContext.")

        # Orchestrator always provides token — None is a calling-code bug.
        # Consistent with the state_id guard above.
        if ctx.token is None:
            raise RuntimeError(f"{self.__class__.__name__} requires token on TransformContext.")
        token_id = ctx.token.token_id

        # 1. Build query from row data
        query_result = self._query_builder.build(row.to_dict())
        if query_result.error is not None:
            self._quarantine_count += 1
            return TransformResult.error(
                query_result.error,
                retryable=False,
            )

        query = query_result.query
        assert query is not None  # guaranteed when error is None

        # 2. Search via provider — Tier 3 boundary (external call)
        self._total_queries += 1
        try:
            chunks = self._provider.search(
                query,
                self._rag_config.top_k,
                self._rag_config.min_score,
                state_id=ctx.state_id,
                token_id=token_id,
            )
        except RetrievalError as e:
            if e.retryable:
                raise  # Engine retry handles transient failures
            self._quarantine_count += 1
            retrieval_error_reason: TransformErrorReason = {
                "reason": "retrieval_failed",
                "error": str(e),
                "cause": f"{type(e).__name__}: {e.__cause__}" if e.__cause__ else str(e),
                "provider": self._rag_config.provider,
            }
            if e.status_code is not None:
                retrieval_error_reason["status_code"] = e.status_code
            return TransformResult.error(retrieval_error_reason, retryable=False)

        # Providers surface skip evidence on the instance so the transform can
        # persist why candidate hits were rejected even when no usable chunks remain.
        skipped_count = self._provider.last_skipped_count
        skipped_reasons = self._provider.last_skipped_reasons

        # 3. Handle zero results
        if not chunks:
            if self._rag_config.on_no_results == "quarantine":
                self._quarantine_count += 1
                no_results_error_reason: TransformErrorReason = {
                    "reason": "no_results",
                    "query": query,
                    "provider": self._rag_config.provider,
                }
                if skipped_count > 0:
                    no_results_error_reason["skipped_count"] = skipped_count
                if skipped_reasons:
                    no_results_error_reason["skipped_reasons"] = skipped_reasons
                return TransformResult.error(no_results_error_reason, retryable=False)
            # on_no_results == "continue" — None sentinels preserve semantic
            # distinction: None means "no retrieval happened", 0.0/"" would
            # fabricate a result indistinguishable from "zero relevance".
            output = row.to_dict()
            output[self._field_context] = None
            output[self._field_score] = None
            output[self._field_count] = 0
            output[self._field_sources] = json.dumps({"v": 1, "sources": []})

            output_contract = propagate_contract(
                input_contract=row.contract,
                output_row=output,
                transform_adds_fields=True,
            )
            output_contract = self._align_output_contract(output_contract)
            no_results_success_metadata: dict[str, Any] = {"chunk_count": 0, "no_results": True}
            if skipped_count > 0:
                no_results_success_metadata["skipped_count"] = skipped_count
            if skipped_reasons:
                no_results_success_metadata["skipped_reasons"] = skipped_reasons
            return TransformResult.success(
                PipelineRow(output, output_contract),
                success_reason={
                    "action": "rag_retrieval",
                    "metadata": no_results_success_metadata,
                },
            )

        # 4. Format context
        self._total_chunks += len(chunks)
        best_score = chunks[0].score  # chunks are ordered by descending score
        self._update_score_stats(best_score)

        formatted = format_context(
            chunks,
            format_mode=self._rag_config.context_format,
            separator=self._rag_config.context_separator,
            max_length=self._rag_config.max_context_length,
        )

        # 5. Build sources envelope
        sources_envelope = {
            "v": 1,
            "sources": [
                {
                    "source_id": chunk.source_id,
                    "score": chunk.score,
                    "metadata": deep_thaw(chunk.metadata),
                }
                for chunk in chunks
            ],
        }

        # 6. Build output row
        output = row.to_dict()
        output[self._field_context] = formatted.text
        output[self._field_score] = best_score
        output[self._field_count] = len(chunks)
        output[self._field_sources] = json.dumps(sources_envelope)

        output_contract = propagate_contract(
            input_contract=row.contract,
            output_row=output,
            transform_adds_fields=True,
        )
        output_contract = self._align_output_contract(output_contract)

        success_metadata: dict[str, Any] = {
            "chunk_count": len(chunks),
            "best_score": best_score,
            "truncated": formatted.truncated,
        }
        if skipped_count > 0:
            success_metadata["skipped_count"] = skipped_count
        if skipped_reasons:
            success_metadata["skipped_reasons"] = skipped_reasons

        return TransformResult.success(
            PipelineRow(output, output_contract),
            success_reason={
                "action": "rag_retrieval",
                "metadata": success_metadata,
            },
        )

    def _configured_collection_name(self, provider_config: Any) -> str:
        """Extract the configured collection/index name for readiness audit records."""
        for field_name in ("collection", "index"):
            value = getattr(provider_config, field_name, None)
            if isinstance(value, str) and value:
                return value
        raise FrameworkBugError(
            f"{self.__class__.__name__} provider config {type(provider_config).__name__} "
            "must expose a non-empty 'collection' or 'index' for readiness auditing."
        )

    def _record_readiness_check(
        self,
        ctx: LifecycleContext,
        *,
        collection: str,
        reachable: bool,
        count: int | None,
        message: str,
    ) -> None:
        """Persist retrieval readiness facts when the audit writer is available."""
        if ctx.landscape is not None:
            ctx.landscape.record_readiness_check(
                run_id=ctx.run_id,
                name=self.name,
                collection=collection,
                reachable=reachable,
                count=count,
                message=message,
            )

    def on_complete(self, ctx: LifecycleContext) -> None:
        """Emit telemetry with run statistics."""
        score_std = 0.0
        if self._score_count >= 2:
            score_std = math.sqrt(self._score_m2 / (self._score_count - 1))

        payload: dict[str, Any] = {
            "event": "rag_retrieval_complete",
            "run_id": self._run_id,
            "provider": self._rag_config.provider,
            "total_queries": self._total_queries,
            "total_chunks": self._total_chunks,
            "quarantine_count": self._quarantine_count,
            "score_mean": self._score_mean if self._score_count > 0 else None,
            "score_std": score_std if self._score_count >= 2 else None,
        }
        self._telemetry_emit(payload)

    def close(self) -> None:
        """Release provider and query builder resources."""
        # Provider may be None if close() is called before on_start() —
        # this is a valid lifecycle path (e.g., config validation failure
        # before the pipeline starts). But if on_start() was called, the
        # provider must exist — that's guaranteed by on_start's construction.
        if self._provider is not None:
            self._provider.close()
        self._query_builder.close()

    def _update_score_stats(self, score: float) -> None:
        """Welford online algorithm for running mean and variance."""
        self._score_count += 1
        delta = score - self._score_mean
        self._score_mean += delta / self._score_count
        delta2 = score - self._score_mean
        self._score_m2 += delta * delta2
