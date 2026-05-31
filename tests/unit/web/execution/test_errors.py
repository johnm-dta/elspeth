"""Tests for execution-layer error types."""

from __future__ import annotations

import pytest

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
from elspeth.web.execution.errors import (
    ExecuteRequestValidationError,
    SemanticContractViolationError,
    UnresolvedInterpretationPlaceholderError,
)


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


class TestUnresolvedInterpretationPlaceholderError:
    """F-17 / F-21 — typed exception for the runtime placeholder gate.

    The exception MUST carry ``(node_id, term)`` tuples (not the
    ``prompt_template`` string — PII risk), MUST refuse construction
    with an empty placeholder list (control-flow bug), and MUST NOT
    subclass ``ExecuteRequestValidationError`` (which maps to 400 in
    the route catch order; this site maps to 422 to mirror the
    ``SemanticContractViolationError`` precedent).
    """

    def test_carries_placeholder_tuples(self) -> None:
        placeholders = (("rate_node", "cool"), ("summarise_node", "important"))
        exc = UnresolvedInterpretationPlaceholderError(placeholders=placeholders)
        assert exc.placeholders == placeholders

    def test_str_names_every_unresolved_site(self) -> None:
        exc = UnresolvedInterpretationPlaceholderError(
            placeholders=(("rate_node", "cool"), ("summarise_node", "important")),
        )
        message = str(exc)
        assert "{{interpretation:cool}}" in message
        assert "rate_node" in message
        assert "{{interpretation:important}}" in message
        assert "summarise_node" in message
        # Operator-actionable: the message MUST tell the operator HOW
        # to resolve the gate, not just WHAT failed.
        assert "request_interpretation_review" in message

    def test_rejects_empty_placeholders_tuple(self) -> None:
        """Offensive guard — an empty placeholders tuple is a caller bug."""
        with pytest.raises(ValueError, match="at least one"):
            UnresolvedInterpretationPlaceholderError(placeholders=())

    def test_is_not_execute_request_validation_error(self) -> None:
        """Route handler maps this to 422, NOT the 400 path that
        ``ExecuteRequestValidationError`` maps to.  Subclassing the wrong
        base would silently demote the status."""
        exc = UnresolvedInterpretationPlaceholderError(
            placeholders=(("rate_node", "cool"),),
        )
        assert not isinstance(exc, ExecuteRequestValidationError)
        # Also NOT a ValueError — the bare ``except ValueError`` branch
        # in the route maps to 404 (state-not-found), which would also
        # demote this status.
        assert not isinstance(exc, ValueError)
