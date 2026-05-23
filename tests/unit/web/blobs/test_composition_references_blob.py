"""Tests for active-run blob reference discovery across composition state."""

from __future__ import annotations

import pytest

from elspeth.contracts.errors import AuditIntegrityError
from elspeth.web.blobs.service import _composition_references_blob


def test_composition_references_blob_finds_transform_inline_content_ref() -> None:
    composition_state = {
        "sources": {"primary": {"plugin": "csv", "options": {"path": "data.csv"}}},
        "transforms": [
            {
                "name": "classify",
                "plugin": "llm",
                "options": {
                    "system_prompt": {
                        "blob_ref": "blob-123",
                        "mode": "inline_content",
                        "sha256": "a" * 64,
                    }
                },
            }
        ],
        "sinks": {"output": {"plugin": "csv", "options": {"path": "out.csv"}}},
    }

    assert _composition_references_blob(composition_state, "blob-123", "/unused")


def test_composition_references_blob_preserves_legacy_path_match() -> None:
    composition_state = {
        "sources": {"primary": {"plugin": "csv", "options": {"path": "/blob/storage.csv"}}},
    }

    assert _composition_references_blob(composition_state, "other-blob", "/blob/storage.csv")


def test_composition_references_blob_crashes_on_corrupt_source_options() -> None:
    composition_state = {
        "sources": {"primary": {"plugin": "csv", "options": ["not", "a", "dict"]}},
    }

    with pytest.raises(AuditIntegrityError, match=r"sources\['primary'\]\.options"):
        _composition_references_blob(composition_state, "blob-123", "/unused")
