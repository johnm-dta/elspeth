"""Tests for LLM transform success_reason audit metadata.

Validates that success_reason captures enough semantic context for
an auditor to understand what the transform decided, even after
payload purge deletes the output row data.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock

import pytest

from elspeth.contracts.engine import BufferEntry
from elspeth.contracts.results import TransformResult
from elspeth.contracts.schema_contract import PipelineRow, SchemaContract
from elspeth.contracts.token_usage import TokenUsage
from elspeth.plugins.transforms.llm.multi_query import QuerySpec
from elspeth.plugins.transforms.llm.provider import FinishReason, LLMProvider, LLMQueryResult
from elspeth.plugins.transforms.llm.templates import PromptTemplate
from elspeth.plugins.transforms.llm.transform import MultiQueryStrategy, SingleQueryStrategy
from elspeth.testing import make_pipeline_row

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _TokenDouble:
    token_id: str


@dataclass(slots=True)
class _TransformContextDouble:
    state_id: str
    run_id: str
    token: _TokenDouble
    shutdown_event: None = None


class _TracerDouble:
    def record_success(self, **_kwargs: Any) -> None:
        return None

    def record_error(self, **_kwargs: Any) -> None:
        return None


class _PooledExecutorDouble:
    def __init__(
        self,
        execute_batch: Callable[..., list[BufferEntry[TransformResult]]],
    ) -> None:
        self.execute_batch = execute_batch


def _make_ctx() -> _TransformContextDouble:
    """Minimal TransformContext double for LLM strategy execution."""
    return _TransformContextDouble(
        state_id="state-123",
        run_id="run-123",
        token=_TokenDouble(token_id="token-1"),
    )


def _make_mock_provider(responses: list[dict[str, Any]] | None = None) -> Mock:
    """Create a mock LLM provider returning JSON responses.

    If responses is None, returns a default success response for every call.
    Otherwise cycles through the provided responses.
    """
    import itertools

    mock_provider = Mock(spec=LLMProvider)

    if responses is None:
        responses = [{"score": 85, "rationale": "Good"}]

    cycle = itertools.cycle(responses)

    def execute_from_list(
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int | None,
        state_id: str,
        token_id: str,
        response_format: dict[str, Any] | None = None,
    ) -> LLMQueryResult:
        return LLMQueryResult(
            content=json.dumps(next(cycle)),
            usage=TokenUsage.known(10, 5),
            model="gpt-4o",
            finish_reason=FinishReason.STOP,
        )

    mock_provider.execute_query.side_effect = execute_from_list
    return mock_provider


def _identity_contract(contract: SchemaContract) -> SchemaContract:
    """Mirror the no-op contract alignment used in these unit tests."""
    return contract


def _identity_row(row: PipelineRow) -> PipelineRow:
    """Mirror the no-op row alignment used in these unit tests."""
    return row


def _make_multi_query_strategy(*, executor: _PooledExecutorDouble | None = None) -> MultiQueryStrategy:
    """Build a MultiQueryStrategy with two simple query specs."""
    specs = [
        QuerySpec(
            name="sentiment",
            input_fields=MappingProxyType({"text": "text"}),
        ),
        QuerySpec(
            name="topic",
            input_fields=MappingProxyType({"text": "text"}),
        ),
    ]
    return MultiQueryStrategy(
        query_specs=specs,
        template=PromptTemplate("Analyze: {{ row.text }}"),
        system_prompt=None,
        system_prompt_source=None,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=None,
        response_field="llm_response",
        align_output_contract=_identity_contract,
        align_output_row_contract=_identity_row,
        executor=executor,
    )


# ---------------------------------------------------------------------------
# Single-query fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def single_query_result() -> TransformResult:
    """Execute SingleQueryStrategy with mocked provider, return the result."""
    strategy = SingleQueryStrategy(
        template=PromptTemplate("Classify: {{ row.text }}"),
        system_prompt=None,
        system_prompt_source=None,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=None,
        response_field="llm_response",
        align_output_contract=_identity_contract,
    )

    mock_provider = Mock(spec=LLMProvider)
    mock_provider.execute_query.return_value = LLMQueryResult(
        content="The analysis is positive.",
        usage=TokenUsage.known(prompt_tokens=10, completion_tokens=5),
        model="gpt-4o",
        finish_reason=FinishReason.STOP,
    )

    row = make_pipeline_row({"text": "hello"})
    ctx = _make_ctx()

    return strategy.execute(row, ctx, provider=mock_provider, tracer=_TracerDouble())


# ---------------------------------------------------------------------------
# Multi-query fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def multi_query_result() -> TransformResult:
    """Execute MultiQueryStrategy sequential path, return the result."""
    strategy = _make_multi_query_strategy(executor=None)
    provider = _make_mock_provider(
        [
            {"score": 85, "rationale": "Good"},
            {"category": "tech", "confidence": 0.9},
        ]
    )
    row = make_pipeline_row({"text": "hello"})
    ctx = _make_ctx()

    return strategy.execute(row, ctx, provider=provider, tracer=_TracerDouble())


@pytest.fixture()
def parallel_multi_query_result() -> TransformResult:
    """Execute MultiQueryStrategy parallel path via mocked PooledExecutor.

    The mock execute_batch calls process_fn for each context, mirroring
    the real PooledExecutor contract.  This lets _process_fn populate the
    audit_metadata_by_index side-channel that _execute_parallel validates.
    """
    from elspeth.plugins.infrastructure.pooling.executor import RowContext

    def _fake_execute_batch(
        contexts: list[RowContext],
        process_fn: Callable[[dict[str, Any], str], TransformResult],
        *,
        shutdown_event: Any | None = None,
    ) -> list[BufferEntry[TransformResult]]:
        entries: list[BufferEntry[TransformResult]] = []
        for i, ctx in enumerate(contexts):
            result = process_fn(ctx.row, ctx.state_id)
            entries.append(
                BufferEntry(
                    submit_index=i,
                    complete_index=i,
                    result=result,
                    submit_timestamp=0.0,
                    complete_timestamp=0.001,
                    buffer_wait_ms=0.0,
                )
            )
        return entries

    executor = _PooledExecutorDouble(_fake_execute_batch)

    strategy = _make_multi_query_strategy(executor=executor)
    provider = _make_mock_provider(
        [
            {"score": 85, "rationale": "Good"},
            {"category": "tech", "confidence": 0.9},
        ]
    )
    row = make_pipeline_row({"text": "hello"})
    ctx = _make_ctx()

    return strategy.execute(row, ctx, provider=provider, tracer=_TracerDouble())


# ---------------------------------------------------------------------------
# Single-query tests
# ---------------------------------------------------------------------------


class TestSingleQuerySuccessReason:
    """SingleQueryStrategy must capture response field and model in success_reason."""

    def test_success_reason_includes_model(
        self,
        single_query_result: TransformResult,
    ) -> None:
        """success_reason must include which model produced the result."""
        assert single_query_result.success_reason is not None
        assert "model" in single_query_result.success_reason.get("metadata", {}), (
            "success_reason.metadata must include 'model' so the audit trail records which model produced the classification"
        )

    def test_success_reason_includes_completion_tokens(
        self,
        single_query_result: TransformResult,
    ) -> None:
        """success_reason must include completion token count for cost attribution."""
        assert single_query_result.success_reason is not None
        metadata = single_query_result.success_reason.get("metadata", {})
        assert "completion_tokens" in metadata, (
            "success_reason.metadata must include 'completion_tokens' for cost attribution in the audit trail"
        )

    def test_success_reason_includes_fields_added(
        self,
        single_query_result: TransformResult,
    ) -> None:
        """success_reason must include which fields were added."""
        assert single_query_result.success_reason is not None
        assert "fields_added" in single_query_result.success_reason
        assert isinstance(single_query_result.success_reason["fields_added"], list)
        assert len(single_query_result.success_reason["fields_added"]) > 0

    def test_success_reason_contains_audit_metadata(
        self,
        single_query_result: TransformResult,
    ) -> None:
        """Audit provenance fields live in success_reason['metadata'], not the row."""
        assert single_query_result.success_reason is not None
        metadata = single_query_result.success_reason["metadata"]
        response_field = "llm_response"
        assert f"{response_field}_template_hash" in metadata
        assert f"{response_field}_variables_hash" in metadata
        assert f"{response_field}_template_source" in metadata
        assert f"{response_field}_lookup_hash" in metadata
        assert f"{response_field}_lookup_source" in metadata
        assert f"{response_field}_system_prompt_source" in metadata

    def test_audit_fields_not_in_row(
        self,
        single_query_result: TransformResult,
    ) -> None:
        """Audit provenance fields must NOT be in the pipeline row."""
        assert single_query_result.row is not None
        row_data = single_query_result.row.to_dict()
        response_field = "llm_response"
        from elspeth.plugins.transforms.llm import LLM_AUDIT_SUFFIXES

        for suffix in LLM_AUDIT_SUFFIXES:
            assert f"{response_field}{suffix}" not in row_data, (
                f"Audit field '{response_field}{suffix}' found in row — should be in success_reason['metadata'] only"
            )


# ---------------------------------------------------------------------------
# Multi-query tests (sequential path)
# ---------------------------------------------------------------------------


class TestMultiQuerySuccessReason:
    """MultiQueryStrategy sequential path must capture model and field info."""

    def test_success_reason_includes_queries_completed(
        self,
        multi_query_result: TransformResult,
    ) -> None:
        """success_reason must include total query count."""
        assert multi_query_result.success_reason is not None
        assert "queries_completed" in multi_query_result.success_reason

    def test_success_reason_includes_model_for_multi_query(
        self,
        multi_query_result: TransformResult,
    ) -> None:
        """success_reason must include model name."""
        assert multi_query_result.success_reason is not None
        metadata = multi_query_result.success_reason.get("metadata", {})
        assert "model" in metadata

    def test_success_reason_includes_fields_added(
        self,
        multi_query_result: TransformResult,
    ) -> None:
        """success_reason must include which fields were added by the queries."""
        assert multi_query_result.success_reason is not None
        assert "fields_added" in multi_query_result.success_reason
        assert isinstance(multi_query_result.success_reason["fields_added"], list)
        assert len(multi_query_result.success_reason["fields_added"]) > 0

    def test_success_reason_contains_audit_metadata(
        self,
        multi_query_result: TransformResult,
    ) -> None:
        """Multi-query audit provenance fields live in success_reason['metadata']."""
        assert multi_query_result.success_reason is not None
        metadata = multi_query_result.success_reason["metadata"]
        for query_name in ("sentiment", "topic"):
            prefix = f"{query_name}_llm_response"
            assert f"{prefix}_template_hash" in metadata
            assert f"{prefix}_variables_hash" in metadata


# ---------------------------------------------------------------------------
# Multi-query tests (parallel path)
# ---------------------------------------------------------------------------


class TestMultiQuerySuccessReasonParallel:
    """MultiQueryStrategy parallel path must capture model and field info."""

    def test_success_reason_includes_queries_completed(
        self,
        parallel_multi_query_result: TransformResult,
    ) -> None:
        """success_reason must include total query count."""
        assert parallel_multi_query_result.success_reason is not None
        assert "queries_completed" in parallel_multi_query_result.success_reason

    def test_success_reason_includes_model_for_parallel(
        self,
        parallel_multi_query_result: TransformResult,
    ) -> None:
        """success_reason must include model name even for parallel path."""
        assert parallel_multi_query_result.success_reason is not None
        metadata = parallel_multi_query_result.success_reason.get("metadata", {})
        assert "model" in metadata

    def test_success_reason_includes_fields_added(
        self,
        parallel_multi_query_result: TransformResult,
    ) -> None:
        """success_reason must include which fields were added."""
        assert parallel_multi_query_result.success_reason is not None
        assert "fields_added" in parallel_multi_query_result.success_reason
        assert isinstance(parallel_multi_query_result.success_reason["fields_added"], list)
        assert len(parallel_multi_query_result.success_reason["fields_added"]) > 0
