"""Tests for execution-layer error types."""

from __future__ import annotations

from elspeth.contracts.plugin_semantics import (
    ContentKind,
    FieldSemanticFacts,
    FieldSemanticRequirement,
    SemanticEdgeContract,
    SemanticOutcome,
    TextFraming,
    UnknownSemanticPolicy,
)
from elspeth.web.composer.state import ValidationEntry
from elspeth.web.execution.errors import SemanticContractViolationError


def _entry(node_id: str = "x") -> ValidationEntry:
    return ValidationEntry(f"node:{node_id}", "msg", "high")


def _contract() -> SemanticEdgeContract:
    facts = FieldSemanticFacts(
        field_name="c",
        content_kind=ContentKind.PLAIN_TEXT,
        text_framing=TextFraming.COMPACT,
        fact_code="t.c.compact",
    )
    req = FieldSemanticRequirement(
        field_name="c",
        accepted_content_kinds=frozenset({ContentKind.PLAIN_TEXT}),
        accepted_text_framings=frozenset({TextFraming.NEWLINE_FRAMED}),
        requirement_code="t.c.req",
        unknown_policy=UnknownSemanticPolicy.FAIL,
    )
    return SemanticEdgeContract(
        from_id="a",
        to_id="b",
        consumer_plugin="line_explode",
        producer_plugin="web_scrape",
        producer_field="c",
        consumer_field="c",
        producer_facts=facts,
        requirement=req,
        outcome=SemanticOutcome.CONFLICT,
    )


class TestSemanticContractViolationError:
    def test_carries_structured_payload(self) -> None:
        entries = (_entry("x"),)
        contracts = (_contract(),)
        exc = SemanticContractViolationError(
            entries=entries,
            contracts=contracts,
        )
        assert exc.entries == entries
        assert exc.contracts == contracts

    def test_str_summarizes_entries(self) -> None:
        entries = (
            _entry("x"),
            ValidationEntry("node:y", "second message", "high"),
        )
        exc = SemanticContractViolationError(entries=entries, contracts=())
        message = str(exc)
        assert "msg" in message
        assert "second message" in message

    def test_is_value_error_subclass_for_existing_callers(self) -> None:
        # /execute callers that catch ValueError must continue to work
        # during migration. Subclass of ValueError keeps that contract.
        exc = SemanticContractViolationError(entries=(_entry(),), contracts=())
        assert isinstance(exc, ValueError)
