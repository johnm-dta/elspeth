"""Tier-3 boundary tests for the composer state-route helpers.

``_source_options_reference_blob_storage`` reads web-authored source options:
missing or non-string path values are skipped, but a hostile string value
propagates ``resolve_data_path``'s failure (e.g. an embedded NUL byte) rather
than being coerced — the ``@trust_boundary`` honesty test below pins that.
"""

from __future__ import annotations

import pytest

from elspeth.web.sessions.routes.composer.state import _source_options_reference_blob_storage


def test_source_options_reference_blob_storage_raises_on_nul_byte_path(tmp_path) -> None:
    with pytest.raises(ValueError):
        _source_options_reference_blob_storage({"path": "blobs/x\x00y"}, data_dir=str(tmp_path))


def test_source_options_reference_blob_storage_skips_non_string_values(tmp_path) -> None:
    assert _source_options_reference_blob_storage({"path": 7, "file": None}, data_dir=str(tmp_path)) is False


def test_reject_unbound_blob_storage_sources_raises_400_on_unbound_blob_path(tmp_path) -> None:
    """A source pointing into session blob storage without a blob_ref binding is rejected."""
    from fastapi import HTTPException

    from elspeth.web.composer.state import CompositionState, PipelineMetadata, SourceSpec
    from elspeth.web.sessions.routes.composer.state import _reject_unbound_blob_storage_sources

    state = CompositionState(
        sources={
            "source": SourceSpec(
                plugin="csv",
                on_success="main",
                options={"path": str(tmp_path / "blobs" / "x.csv")},
                on_validation_failure="discard",
            )
        },
        nodes=(),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )
    with pytest.raises(HTTPException) as exc_info:
        _reject_unbound_blob_storage_sources(state, data_dir=str(tmp_path))
    assert exc_info.value.status_code == 400


def _llm_node_state_with_requirements(requirements: list) -> object:
    from elspeth.web.composer.state import CompositionState, NodeSpec, PipelineMetadata

    return CompositionState(
        sources={},
        nodes=(
            NodeSpec(
                id="score",
                node_type="transform",
                plugin="llm",
                input="source",
                on_success="main",
                on_error="discard",
                options={
                    "model": "anthropic/claude-haiku-4.5",
                    "prompt_template": "Score this: {{ row.value }}",
                    "interpretation_requirements": requirements,
                },
                condition=None,
                routes=None,
                fork_to=None,
                branches=None,
                policy=None,
                merge=None,
            ),
        ),
        edges=(),
        outputs=(),
        metadata=PipelineMetadata(),
        version=1,
    )


def test_reject_malformed_interpretation_requirements_raises_400() -> None:
    """Hand-written requirement rows the interpretation schema would refuse
    are rejected with HTTP 400 BEFORE persistence (elspeth-ae5160c3cb), and
    the error names the node without echoing row content."""
    from fastapi import HTTPException

    from elspeth.web.sessions.routes.composer.state import _reject_malformed_interpretation_requirements

    state = _llm_node_state_with_requirements([{"kind": "not_a_kind", "user_term": "x", "status": "pending"}])
    with pytest.raises(HTTPException) as exc_info:
        _reject_malformed_interpretation_requirements(state)
    assert exc_info.value.status_code == 400
    assert "score" in exc_info.value.detail
    assert "not_a_kind" not in exc_info.value.detail


def test_reject_malformed_interpretation_requirements_passes_staged_rows() -> None:
    """Rows shaped like the importer's own auto-stagers pass the gate."""
    from elspeth.web.sessions.routes.composer.state import _reject_malformed_interpretation_requirements

    state = _llm_node_state_with_requirements(
        [
            {
                "id": "prompt_template_review:score",
                "kind": "llm_prompt_template",
                "user_term": "llm_prompt_template:score",
                "status": "pending",
                "draft": "Score this: {{ row.value }}",
            }
        ]
    )
    _reject_malformed_interpretation_requirements(state)


@pytest.mark.asyncio
async def test_surface_imported_reviews_skips_writer_rejected_rows() -> None:
    """The interpretation-event writer boundary rejects a mismatched
    requirement row by raising ``ValueError`` — pinned with ``pytest.raises``
    — and the import surfacer converts that raise into a fail-closed SKIP
    (the state already persisted; a propagated raise would 500 the import).
    Resolved rows and pending rows without a string draft never reach the
    writer; only surviving pending rows surface (elspeth-ae5160c3cb)."""
    from uuid import uuid4

    from elspeth.web.sessions.routes.composer.state import _surface_imported_interpretation_review_events

    surfaced: list[dict] = []

    class _Service:
        async def create_pending_interpretation_event(self, **kwargs):
            if kwargs["user_term"] == "llm_model_choice:score":
                raise ValueError("writer boundary rejected the requirement binding")
            surfaced.append(kwargs)

    service = _Service()
    with pytest.raises(ValueError):
        await service.create_pending_interpretation_event(user_term="llm_model_choice:score")

    state = _llm_node_state_with_requirements(
        [
            # Well-formed pending row: surfaces.
            {
                "id": "prompt_template_review:score",
                "kind": "llm_prompt_template",
                "user_term": "llm_prompt_template:score",
                "status": "pending",
                "draft": "Score this: {{ row.value }}",
            },
            # Well-formed pending row the writer boundary rejects: skipped
            # fail-closed, never propagated.
            {
                "id": "model_choice_review:score",
                "kind": "llm_model_choice",
                "user_term": "llm_model_choice:score",
                "status": "pending",
                "draft": "anthropic/claude-haiku-4.5",
            },
            # Pending row without a string draft: never reaches the writer.
            {
                "id": "vague:score",
                "kind": "vague_term",
                "user_term": "cool",
                "status": "pending",
                "draft": None,
            },
            # Resolved row: never reaches the writer.
            {
                "id": "resolved:score",
                "kind": "llm_prompt_template",
                "user_term": "old",
                "status": "resolved",
                "draft": "old draft",
                "accepted_value": "old draft",
            },
        ]
    )

    await _surface_imported_interpretation_review_events(
        service,  # type: ignore[arg-type]
        session_id=uuid4(),
        state=state,
        composition_state_id=uuid4(),
    )

    assert [call["user_term"] for call in surfaced] == ["llm_prompt_template:score"]
    assert surfaced[0]["llm_draft"] == "Score this: {{ row.value }}"
    assert surfaced[0]["model_identifier"] == "yaml_import"
