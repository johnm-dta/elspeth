"""Redaction tests for ``request_interpretation_review`` (Phase 5b Task 5).

Three tests cover the spec at lines 2498-2504:

1. The redaction model accepts a valid argument dict (Pydantic structural).
2. The summariser truncates ``user_term`` and ``llm_draft`` to 64 chars + "…".
3. The audit envelope canonicalises the redacted form (the tool-call column
   carries the summarised shape, not the raw user content).

Spec ref: docs/composer/ux-redesign-2026-05/18a-phase-5b-backend.md §"Redaction model".
"""

from __future__ import annotations

import pytest

from elspeth.web.composer.redaction import (
    MANIFEST,
    _RequestInterpretationReviewRedactionModel,
    _summarize_interpretation_term,
    redact_tool_call_arguments,
)
from elspeth.web.composer.redaction_telemetry import RedactionTelemetry


class _NullTelemetry(RedactionTelemetry):
    """Test double — records nothing, raises nothing."""

    def manifest_dispatch(self, *, tool_name: str, shape: str) -> None:
        return None

    def summarizer_error(self, *, tool_name: str) -> None:
        return None

    def unknown_response_key(self, *, tool_name: str) -> None:
        return None


def test_redaction_model_accepts_valid_argument_dict() -> None:
    """Spec test 1: structural acceptance of a valid argument shape."""
    model = _RequestInterpretationReviewRedactionModel.model_validate(
        {
            "affected_node_id": "rate_node",
            "kind": "vague_term",
            "user_term": "cool",
            "llm_draft": "Visually appealing and well organised.",
        }
    )
    assert model.affected_node_id == "rate_node"
    assert model.kind == "vague_term"
    assert model.user_term == "cool"
    assert model.llm_draft == "Visually appealing and well organised."


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("cool", "<interpretation-term:4-chars>"),
        ("", "<interpretation-term:0-chars>"),
        ("a" * 64, "<interpretation-term:64-chars>"),
        ("a" * 65, "<interpretation-term:65-chars:truncated>"),
        ("a" * 200, "<interpretation-term:200-chars:truncated>"),
    ],
)
def test_summarize_fixed_form_with_length_disclosure(raw: str, expected: str) -> None:
    """Spec test 2 (part 1): the summariser collapses to a fixed-form
    ``<interpretation-term:N-chars[:truncated]>`` scalar at every reachable
    input including the empty string — this is what makes
    ``redacted_value != raw_value`` uniformly true for the redaction-
    completeness property test."""
    assert _summarize_interpretation_term(raw) == expected


def test_audit_envelope_carries_redacted_form() -> None:
    """Spec test 3: redact_tool_call_arguments substitutes the summarised
    form into the audit row's tool-call column.

    ``user_term`` and ``llm_draft`` collapse to the fixed-form
    ``<interpretation-term:N-chars[:truncated]>`` scalar; ``affected_node_id``
    flows through verbatim (it's structural metadata, not user content).
    """
    long_term = "very-important-context-string-that-exceeds-the-cap-by-quite-a-lot-of-characters"
    long_draft = "x" * 200
    raw_arguments = {
        "affected_node_id": "rate_node",
        "kind": "vague_term",
        "user_term": long_term,
        "llm_draft": long_draft,
    }
    redacted = redact_tool_call_arguments(
        "request_interpretation_review",
        raw_arguments,
        telemetry=_NullTelemetry(),
    )

    # affected_node_id flows through unchanged.
    assert redacted["affected_node_id"] == "rate_node"
    assert redacted["kind"] == "vague_term"
    # user_term and llm_draft are summarised to the fixed-form scalar.
    assert redacted["user_term"] == f"<interpretation-term:{len(long_term)}-chars:truncated>"
    assert redacted["llm_draft"] == f"<interpretation-term:{len(long_draft)}-chars:truncated>"
    # Sanity: raw values did NOT leak through.
    assert long_term not in redacted["user_term"]
    assert long_draft not in redacted["llm_draft"]


def test_manifest_entry_is_type_driven() -> None:
    """MANIFEST registration: the new tool uses the type-driven shape
    (argument_model) so the walker discovers Sensitive paths from the
    Pydantic schema rather than relying on a declarative policy."""
    entry = MANIFEST["request_interpretation_review"]
    assert entry.argument_model is _RequestInterpretationReviewRedactionModel
    assert entry.policy is None
