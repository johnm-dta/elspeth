"""Operator documentation contract for epoch-28 sink effects."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[3]
RUNBOOK = ROOT / "docs/runbooks/sink-effect-recovery.md"
ARCHITECTURE = ROOT / "docs/architecture/token-scheduler-state-engine.md"
PROTOCOL = ROOT / "docs/contracts/plugin-protocol.md"
CONFIGURATION = ROOT / "docs/reference/configuration.md"
RELEASE = ROOT / "docs/release/README.md"

REQUIRED_EFFECT_DOC_TERMS = {
    "RESERVED -> PREPARED -> IN_FLIGHT -> FINALIZED",
    "NOT_APPLIED",
    "APPLIED_WITH_EXACT_DESCRIPTOR",
    "UNKNOWN",
    "NO_INSPECTION_REQUIRED",
    "publication_performed",
    "epoch 28",
}


def test_sink_effect_runbook_contains_complete_recovery_contract() -> None:
    content = RUNBOOK.read_text(encoding="utf-8")
    missing = sorted(term for term in REQUIRED_EFFECT_DOC_TERMS if term not in content)
    assert missing == []
    for topic in (
        "blocked successor",
        "response-lost",
        "staging cleanup",
        "target-side ledger",
        "speculative commit",
        "signer key ID",
    ):
        assert topic in content


def test_sink_effect_docs_link_to_the_runbook_and_release_index() -> None:
    assert "../runbooks/sink-effect-recovery.md" in ARCHITECTURE.read_text(encoding="utf-8")
    assert "../runbooks/sink-effect-recovery.md" in PROTOCOL.read_text(encoding="utf-8")
    assert "../runbooks/sink-effect-recovery.md" in RELEASE.read_text(encoding="utf-8")


def test_configuration_documents_bounded_effect_settings_and_key_rotation() -> None:
    content = CONFIGURATION.read_text(encoding="utf-8")
    for term in (
        "total_byte_limit",
        "total_record_limit",
        "chunk_limit",
        "per_chunk_byte_limit",
        "per_chunk_record_limit",
        "spool_root",
        "signer_key_id",
        "content_store_id",
    ):
        assert term in content
    assert "low-entropy key-derived identifiers" in content
