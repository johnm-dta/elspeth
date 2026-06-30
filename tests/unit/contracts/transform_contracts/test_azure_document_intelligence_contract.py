"""ADR-009 forward-invariant contract tests for azure_document_intelligence.

The transform is ``passes_through_input=True`` and uses BatchTransformMixin, so it
is covered by the forward invariant (not the backward probe). These tests exercise
the real per-row LRO path via ``execute_forward_invariant_probe`` (which drives
``_process_single_with_state`` with a local probe client), asserting the row is
enriched AND every input field is preserved.
"""

from __future__ import annotations

from unittest.mock import Mock

from elspeth.contracts import Determinism
from elspeth.plugins.transforms.azure.document_intelligence import AzureDocumentIntelligence
from elspeth.testing import make_pipeline_row


def _probe_transform() -> AzureDocumentIntelligence:
    return AzureDocumentIntelligence(AzureDocumentIntelligence.probe_config())


def test_transform_declares_external_call_passthrough() -> None:
    t = _probe_transform()
    assert t.name == "azure_document_intelligence"
    assert t.determinism is Determinism.EXTERNAL_CALL
    assert t.passes_through_input is True
    assert t.input_schema is not None
    assert t.output_schema is not None


def test_forward_invariant_probe_enriches_and_passes_through() -> None:
    t = _probe_transform()
    probe = make_pipeline_row({"existing_field": "keep-me"})
    rows = t.forward_invariant_probe_rows(probe)
    assert len(rows) == 1

    ctx = Mock()
    ctx.state_id = "probe-state"
    ctx.token = None

    result = t.execute_forward_invariant_probe(rows, ctx)

    assert result.status == "success"
    out = result.row.to_dict()
    # Enriched: content_field declared by probe_config is present.
    assert out["di_content"] == "probe"
    # Pass-through: the augmented source field and the pre-existing field survive.
    assert out[t._source_field] == "https://probe.example/doc.pdf"
    assert out["existing_field"] == "keep-me"


def test_forward_invariant_probe_success_reason_action() -> None:
    t = _probe_transform()
    rows = t.forward_invariant_probe_rows(make_pipeline_row({}))
    ctx = Mock()
    ctx.state_id = "probe-state"
    ctx.token = None
    result = t.execute_forward_invariant_probe(rows, ctx)
    assert result.success_reason["action"] == "enriched"
