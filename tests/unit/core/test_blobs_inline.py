"""Inline blob content resolver tests."""

from __future__ import annotations

from uuid import UUID

import pytest

from elspeth.contracts.blobs_inline import BlobContentResolutionError
from elspeth.core.blobs_inline import _discover_blob_content_refs

VALID_HASH = "a" * 64
BLOB1 = "5b7a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3a4b"
BLOB2 = "7c3a4e0e-9e4a-4f0b-8d3e-2c0e1f0d3aaa"


def _marker(blob_id: str = BLOB1, sha256: str = VALID_HASH) -> dict[str, str]:
    return {"blob_ref": blob_id, "mode": "inline_content", "sha256": sha256}


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
