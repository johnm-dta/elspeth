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
