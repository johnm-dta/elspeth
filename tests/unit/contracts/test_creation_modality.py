"""Tests for blob creation-modality contract helpers."""

from __future__ import annotations

from elspeth.contracts.enums import CreationModality, is_llm_authored_creation_modality


def test_creation_modality_exposes_llm_provenance_predicate() -> None:
    """The provenance invariant is discoverable from the enum value."""
    assert not CreationModality.VERBATIM.requires_llm_provenance()
    assert CreationModality.LLM_GENERATED.requires_llm_provenance()
    assert CreationModality.DISAMBIGUATED.requires_llm_provenance()
    assert CreationModality.LLM_GENERATED_THEN_AMENDED.requires_llm_provenance()


def test_public_helper_matches_creation_modality_method() -> None:
    """The legacy helper delegates to the enum-owned invariant."""
    for modality in CreationModality:
        assert is_llm_authored_creation_modality(modality) is modality.requires_llm_provenance()
