"""Inline blob content resolver tests."""

from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from elspeth.contracts.blobs import BlobContentMissingError, BlobIntegrityError, BlobNotFoundError, BlobStateError
from elspeth.contracts.blobs_inline import BlobContentResolutionError, BlobInlineRef
from elspeth.core.blobs_inline import _discover_blob_content_refs, _fetch_blob_contents

VALID_HASH = "a" * 64
BLOB1 = "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b"
BLOB2 = "7c3a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3aaa"


def _marker(blob_id: str = BLOB1, sha256: str = VALID_HASH) -> dict[str, str]:
    return {"blob_ref": blob_id, "mode": "inline_content", "sha256": sha256}


def _ref(blob_id: str = BLOB1, field_path: str = "source.options.x") -> BlobInlineRef:
    return BlobInlineRef(field_path=field_path, blob_id=UUID(blob_id), sha256=VALID_HASH, encoding="utf-8")


class TestDiscoverBlobContentRefs:
    def test_discovers_source_option_refs(self) -> None:
        config = {"source": {"plugin": "csv", "options": {"system_prompt": _marker()}}}

        refs = _discover_blob_content_refs(config)

        assert len(refs) == 1
        assert refs[0].field_path == "source.options.system_prompt"
        assert refs[0].blob_id == UUID(BLOB1)
        assert refs[0].sha256 == VALID_HASH
        assert refs[0].encoding == "utf-8"

    def test_discovers_node_option_refs_across_node_collections(self) -> None:
        config = {
            "transforms": [
                {"name": "classify", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
            ],
            "gates": [
                {"name": "policy-check", "plugin": "expression", "options": {"template": _marker(BLOB2)}},
            ],
            "aggregations": [
                {"name": "rollup", "plugin": "group_by", "options": {}},
            ],
            "coalesce": [
                {"name": "merge", "plugin": "first", "options": {"explain": _marker(BLOB1)}},
            ],
        }

        refs = _discover_blob_content_refs(config)

        assert {ref.field_path for ref in refs} == {
            "node:classify.options.system_prompt",
            "node:policy-check.options.template",
            "node:merge.options.explain",
        }

    def test_discovers_output_and_sink_refs_with_same_canonical_prefix(self) -> None:
        config = {
            "outputs": {
                "state-view": {"plugin": "json", "options": {"body_template": _marker(BLOB1)}},
            },
            "sinks": {
                "writeback": {"plugin": "csv", "options": {"footer_template": _marker(BLOB2)}},
            },
        }

        refs = _discover_blob_content_refs(config)

        assert {ref.field_path for ref in refs} == {
            "output:state-view.options.body_template",
            "output:writeback.options.footer_template",
        }

    def test_discovers_nested_refs_inside_options(self) -> None:
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "auth": {
                        "system_prompt": _marker(),
                    },
                },
            },
        }

        refs = _discover_blob_content_refs(config)

        assert len(refs) == 1
        assert refs[0].field_path == "source.options.auth.system_prompt"

    def test_ignores_bind_source_refs_and_secret_refs(self) -> None:
        config = {
            "source": {
                "plugin": "csv",
                "options": {
                    "blob_ref": BLOB1,
                    "mode": "bind_source",
                    "path": "/tmp/input.csv",
                    "api_key": {"secret_ref": "OPENROUTER_KEY"},
                },
            }
        }

        assert _discover_blob_content_refs(config) == []

    def test_emits_one_ref_per_field_path_even_for_same_blob(self) -> None:
        config = {
            "transforms": [
                {"name": "a", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
                {"name": "b", "plugin": "llm", "options": {"system_prompt": _marker(BLOB1)}},
            ]
        }

        refs = _discover_blob_content_refs(config)

        assert len(refs) == 2
        assert {ref.field_path for ref in refs} == {
            "node:a.options.system_prompt",
            "node:b.options.system_prompt",
        }

    def test_malformed_markers_raise_batched_error(self) -> None:
        config = {"source": {"plugin": "csv", "options": {"system_prompt": {"blob_ref": BLOB1}}}}

        with pytest.raises(BlobContentResolutionError) as exc_info:
            _discover_blob_content_refs(config)

        assert exc_info.value.malformed == (("source.options.system_prompt", "missing mode"),)

    def test_markers_inside_lists_are_malformed_because_paths_would_be_positional(self) -> None:
        config = {"source": {"plugin": "csv", "options": {"prompts": [_marker()]}}}

        with pytest.raises(BlobContentResolutionError) as exc_info:
            _discover_blob_content_refs(config)

        assert exc_info.value.malformed == (("source.options.prompts", "inline blob refs inside lists are not supported"),)


class TestFetchBlobContents:
    @pytest.mark.asyncio
    async def test_returns_bytes_by_ref(self) -> None:
        blob_service = AsyncMock()
        blob_service.read_blob_content.return_value = b"content"
        ref = _ref()

        fetched = await _fetch_blob_contents(blob_service, [ref])

        blob_service.read_blob_content.assert_awaited_once_with(UUID(BLOB1))
        assert fetched == {ref: b"content"}

    @pytest.mark.asyncio
    async def test_dedupes_reads_by_blob_id(self) -> None:
        blob_service = AsyncMock()
        blob_service.read_blob_content.return_value = b"content"
        refs = [
            _ref(BLOB1, "source.options.a"),
            _ref(BLOB1, "source.options.b"),
        ]

        fetched = await _fetch_blob_contents(blob_service, refs)

        blob_service.read_blob_content.assert_awaited_once_with(UUID(BLOB1))
        assert fetched == {refs[0]: b"content", refs[1]: b"content"}

    @pytest.mark.asyncio
    async def test_propagates_integrity_errors(self) -> None:
        blob_service = AsyncMock()
        blob_service.read_blob_content.side_effect = BlobIntegrityError(BLOB1, expected="a" * 64, actual="b" * 64)

        with pytest.raises(BlobIntegrityError):
            await _fetch_blob_contents(blob_service, [_ref()])

    @pytest.mark.asyncio
    async def test_propagates_content_missing_errors(self) -> None:
        blob_service = AsyncMock()
        blob_service.read_blob_content.side_effect = BlobContentMissingError(BLOB1, storage_path="/tmp/blob.csv")

        with pytest.raises(BlobContentMissingError):
            await _fetch_blob_contents(blob_service, [_ref()])

    @pytest.mark.asyncio
    async def test_collects_not_found_by_field_path(self) -> None:
        blob_service = AsyncMock()
        blob_service.read_blob_content.side_effect = BlobNotFoundError(BLOB1)

        with pytest.raises(BlobContentResolutionError) as exc_info:
            await _fetch_blob_contents(blob_service, [_ref()])

        assert exc_info.value.missing == ("source.options.x",)

    @pytest.mark.asyncio
    async def test_collects_not_ready_by_field_path(self) -> None:
        blob_service = AsyncMock()
        blob_service.read_blob_content.side_effect = BlobStateError(BLOB1, message="Blob is pending")

        with pytest.raises(BlobContentResolutionError) as exc_info:
            await _fetch_blob_contents(blob_service, [_ref()])

        assert exc_info.value.not_ready == (("source.options.x", "Blob is pending"),)
